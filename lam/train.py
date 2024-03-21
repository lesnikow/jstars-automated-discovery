# -*- coding: utf-8 -*-
"""train module
Define our train module for our lunar anomalies project.
This module is used to train our VAE model. 
"""

import math
import os
import pickle
import shutil
import sys
import time
from importlib import reload
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms, utils
from tqdm import tqdm

import data
import plotters
import utils
import vae_cnn
import vae_losses
from metrics import KDE_Plot
from metrics import PrecisionRecallCurve, RecieverOperatingCharacteristicCurve
from vae_cnn import VAE_CNN
from vae_losses import LossVAE


def main():
    experiment_user_string = "train_jstars"
    mode = "train"
    load_saved_model = False
    if load_saved_model:
        load_saved_model_fp = os.path.join("models", "model.pt")
    print("\n\nStarting {}...".format(experiment_user_string))

    if mode == "train":
        eps_to_train = 16
    l_space_dim = 2 ** 8
    batch_size = 1024 * 8
    cuda = torch.torch.cuda.is_available()
    print(f"cuda is {cuda}")
    torch.manual_seed(0)
    device = torch.device("cuda" if cuda else "cpu")

    util = utils.Utilitator(verbose=False, mode=mode)
    exp_dir_fp = util.return_experiment_directory_string(experiment_user_string)
    util.make_directories_from_experiment_directory(exp_dir_fp)
    print(f"Experiment directory filepath is: {exp_dir_fp}")

    model = vae_cnn.VAE_CNN(
        experiment_directory=exp_dir_fp, l_space_dim=l_space_dim, verbose=False
    ).to(device)
    loss = vae_losses.LossVAE(
        l1_or_l2_loss="l1",
        include_kld_loss=True,
        loss_lambda=4.0,
        batch_size=batch_size,
        vae_loss_verbose=False,
    )
    opt = optim.Adam(model.parameters(), lr=1e-3)

    if load_saved_model:
        model.load_state_dict(torch.load(load_saved_model_fp))

    if mode == "val":
        val_root_fp = ""
        val_dataset = data.LunarDataset(root=val_root_fp, mode="val", verbose=True)
        val_dataloader = data.LunarDataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )

    if mode == "train":
        train_root_fp = "data/processed/train/"
        train_dataset = data.LunarDataset(
            root=train_root_fp, mode="train", verbose=True
        )
        train_dataloader = data.LunarDataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )

    if mode in ["val", "test"]:
        samples_df = pd.DataFrame(
            data=val_dataset.samples, columns=["sample_filepath", "torch_class_index"]
        )
        samples_df["y_true"] = 1 - samples_df["torch_class_index"]
        samples_df["anomaly_score"] = 0.0

        y_labels, y_scores = model.validate_on_a_val_or_test_set(
            val_or_test_loader=val_dataloader,
            use_reconstruction_loss=True,
            use_distribution_loss=True,
            reconstruction_loss_metric=torch.nn.modules.loss.MSELoss,
            loss_lambda=loss.loss_lambda,
            device=device,
            batch_size=batch_size,
            batch_break_value=8,
            verbose=True,
        )

        pr_curve = PrecisionRecallCurve(
            y_labels,
            y_scores,
            pos_label=0,
            save_to_disk_fp=os.path.join(exp_dir_fp, "curves", "pr_curve"),
        )
        pr_curve.make_plot(show=False)

        roc_curve = RecieverOperatingCharacteristicCurve(
            y_labels,
            y_scores,
            pos_label=0,
            save_to_disk_fp=os.path.join(exp_dir_fp, "curves", "roc_curve"),
        )
        roc_curve.make_plot(show=False)
        
        kde_plot = KDE_Plot(
            y_labels,
            y_scores,
            pos_label=0,
            save_to_disk_fp=os.path.join(exp_dir_fp, "kde_plot"),
        )
        kde_plot.make_plot(show=False)


        samples_df["anomaly_score"][: len(y_scores)] = y_scores
        samples_df = samples_df[: len(y_scores)]
        samples_df = samples_df.sort_values("anomaly_score", ascending=False)
        sorted_samples_fps = pd.Series(samples_df["sample_filepath"]).to_list()

        copy_dry_run = False
        if copy_dry_run:
            print(
                "This is a DRY RUN for copying images into our most_anomalous_samples directory."
            )
        dst_fp_root = os.path.join(exp_dir_fp, "most_anomalous_samples")

        anomaly_score_rank = 0
        for src_fp in sorted_samples_fps[:128]:
            src_file_path_parts = src_fp.split("/")
            lroc_img_id, img_id = src_file_path_parts[-2], src_file_path_parts[-1]
            dst_fp = os.path.join(
                dst_fp_root, f"{anomaly_score_rank:04d}__{lroc_img_id}__{img_id}"
            )

            if copy_dry_run:
                print(f"DRY RUN copy of {src_fp} to {dst_fp}.\n")

            elif not copy_dry_run:
                shutil.copy(src_fp, dst_fp)

            anomaly_score_rank += 1

    elif mode == "train":
        model.train()
        start_ep = model.epoch
        plotter = plotters.LossPlotterBasic(exp_dir_fp)
        torch.save(
            model.state_dict(),
            os.path.join(
                exp_dir_fp, "models", "model_boe_{}.pt".format(str(model.epoch))
            ),
        )
        for epoch in tqdm(range(start_ep, start_ep + eps_to_train)):
            model.train_for_a_single_epoch(
                model.epoch,
                train_loader=train_dataloader,
                optimizer=opt,
                loss_to_use=loss,
                plotter=plotter,
                device=device,
                batch_size=batch_size,
            )
            torch.save(
                model.state_dict(),
                os.path.join(
                    exp_dir_fp,
                    "models",
                    "model_eoe_{}.pt".format(str(max(model.epoch - 1, 0))),
                ),
            )
            plotter.write_out_train_losses_to_disk(model, exp_dir_fp)

        torch.save(
            model.state_dict(),
            os.path.join(
                exp_dir_fp, "models", "model_eot_{}.pt".format(str(model.epoch))
            ),
        )

    print(f"Experiment run finished.")
    print(f"Experiment results may be found in {exp_dir_fp}.\n")


if __name__ == "__main__":
    main()
