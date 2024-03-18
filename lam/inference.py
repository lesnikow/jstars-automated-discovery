#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""inference module
An inference engine for analyzing large sets of raw image data using a
pre-trained variational autoencoder convolutional neural network. Outputs
a .lam file, or local anomaly map file, for use in e.g. out evaluate model.

Example usage:
python inference_module.py \
    --raw_imgs_fp "/path/to/raw/satellite/images" \
    --model_fp "/path/to/pretrained/model/model.pt" \
    --out_fp_base "/path/to/output/directory/for/lam/files" \
    --exp_str "lroc_image_analysis" \
    --batch_size 2048 \
    --loss_lambda 4.0 \
    --seed  \
    --keep_existing_lams \
    --use_distribution_loss
"""

import argparse
import logging
import os
import random
import sys
import time

import numpy as np
from PIL import Image
import torch

from data_inf import InfDataset, InfDataLoader
import vae_cnn

default_args = {
    "raw_imgs_fp": os.path.join("/", "nvme", "raws"),
    "model_fp": os.path.join(
        "models",
        "train1_arts_exp_time_1646954340__stage_iii_train1"
        "_squares_2K_debug_models_model_eoe_5.pt",
    ),
    "out_fp_base": os.path.join("data", "processed"),
    "exp_str": "stage_3_inference",
    "loss_lambda": 4.0,
    "batch_size": 2 ** 11,
    "use_distribution_loss": True,
    "seed": 0,
    "keep_existing_lams": True,
}


def main(
    raw_imgs_fp=default_args["raw_imgs_fp"],
    model_fp=default_args["model_fp"],
    out_fp_base=default_args["out_fp_base"],
    exp_str=default_args["exp_str"],
    batch_size=default_args["batch_size"],
    loss_lambda=default_args["loss_lambda"],
    use_distribution_loss=default_args["use_distribution_loss"],
    seed=default_args["seed"],
    keep_existing_lams=default_args["keep_existing_lams"],
):
    """This module's main method."""

    # Setup
    Image.MAX_IMAGE_PIXELS = 5e8
    torch.manual_seed(default_args["seed"])
    random.seed(default_args["seed"])
    np.random.seed(default_args["seed"])

    exp_dir_fp = os.path.join("results", exp_str + "_time_" + str(int(time.time())))
    os.makedirs(exp_dir_fp)

    cuda = torch.torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    model = vae_cnn.VAE_CNN(
        experiment_directory=exp_dir_fp, l_space_dim=2 ** 8, verbose=False
    ).to(device)
    model.load_state_dict(torch.load(model_fp))

    inf_dataset = InfDataset(root=raw_imgs_fp)
    logging.info(inf_dataset)
    inf_dataloader = InfDataLoader(inf_dataset, shuffle=False)

    # Inference
    model.eval()
    anom_scores = {}
    reconstruction_loss_metric = torch.nn.modules.loss.MSELoss
    if not out_fp_base:
        out_fp_base = os.path.join("data", "processed")
    if not os.path.exists(out_fp_base):
        os.makedirs(out_fp_base)
    with torch.no_grad():
        for lroc_raw_img_idx, (lroc_raw_img, _) in enumerate(inf_dataloader):
            logging.info(
                "Starting inference on %s of %s", lroc_raw_img_idx, len(inf_dataset)
            )
            lroc_id = inf_dataset.samples[lroc_raw_img_idx][0].split("/")[-1]
            lroc_id = lroc_id.split("_")[0]
            logging.info("Working on LROC raw with id: %s", lroc_id)
            out_fp = os.path.join(out_fp_base, lroc_id + ".lam")
            if os.path.exists(out_fp) and keep_existing_lams:
                logging.info(
                    "Out file %s already exists, skipping inference for %s.",
                    out_fp,
                    lroc_id,
                )
                continue

            anom_scores[lroc_id] = []
            ## Tensor formatting
            data = lroc_raw_img.to(device)
            data = torch.squeeze(data)
            height, width = data.shape[0], data.shape[1]

            ### Trim dims to divisible by 64
            y_dim_trim = (data.shape[0] // 64) * 64
            x_dim_trim = (data.shape[1] // 64) * 64
            data = data[:y_dim_trim, :x_dim_trim]
            h_trim, w_trim = data.shape[0], data.shape[1]
            h_trim_lam, w_trim_lam = h_trim / 64, w_trim / 64

            ## Reshape to one long tensor of N x C x H x W
            size, stride = 64, 64
            data = data.unfold(0, size, stride).unfold(1, size, stride)
            data = torch.flatten(data, start_dim=0, end_dim=1)
            data = data[:, None, :, :]

            ## Splice into batches that fit into GPU memory
            total_batches_for_this_lroc_raw = data.shape[0] // batch_size + 1
            for i in range(total_batches_for_this_lroc_raw):
                batch = data[i * batch_size : i * batch_size + batch_size]
                reconstructed_batch, mu, sigma = model.forward(batch)
                reconstruction_loss_batch = reconstruction_loss_metric(
                    reduction="none"
                )(batch, reconstructed_batch)
                reconstruction_loss_batch = torch.squeeze(
                    torch.sum(reconstruction_loss_batch, dim=[2, 3]), -1
                )
                distribution_loss_batch = torch.zeros_like(reconstruction_loss_batch)
                if use_distribution_loss:
                    pow_mu = torch.pow(mu, 2)
                    pow_mu_summed = torch.sum(pow_mu, dim=[1])
                    distribution_loss_batch = pow_mu_summed * loss_lambda
                vae_anomaly_score_batch = torch.add(
                    reconstruction_loss_batch, distribution_loss_batch
                )

                anom_scores[lroc_id].extend(vae_anomaly_score_batch.tolist())
                del batch, reconstructed_batch, mu, sigma
                del reconstruction_loss_batch, distribution_loss_batch
                del pow_mu, pow_mu_summed
                del vae_anomaly_score_batch
                torch.cuda.empty_cache()

            del lroc_raw_img, data
            del y_dim_trim, x_dim_trim, size, stride
            del total_batches_for_this_lroc_raw
            torch.cuda.empty_cache()

            local_anom_map_list = anom_scores[lroc_id]
            with open(out_fp, "w") as output:
                output.write(f"{lroc_id}\n")
                output.write(f"{height}\n{width}\n")
                output.write(f"{h_trim}\n{w_trim}\n")
                output.write(f"{h_trim_lam}\n{w_trim_lam}\n")
                output.write(f"{h_trim_lam * w_trim_lam}\n")
                output.write("# Anomaly scores generated below\n")
                for anom_score in local_anom_map_list:
                    output.write("{:.4f}".format(anom_score))
                    output.write("\n")
            logging.info("Done writing out local anomaly map")
            # os.remove(lroc_raw_img_fp)

    logging.info("anom_scores.keys() is %s", anom_scores.keys())


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f"log_time{time.time()}.log"),
        ],
    )
    logging.info("Starting main block.")

    parser = argparse.ArgumentParser(description="Inference module")
    parser.add_argument(
        "--raw_imgs_fp",
        "-r",
        type=str,
        default=default_args["raw_imgs_fp"],
        help=f"Path to raw LROC images. Default: \"{default_args['raw_imgs_fp']}\"",
    )
    parser.add_argument(
        "--model_fp",
        "-m",
        type=str,
        default=default_args["model_fp"],
        help=f"Path to trained model. Default: \"{default_args['model_fp']}\"",
    )
    parser.add_argument(
        "--out_fp_base",
        "-o",
        type=str,
        default=default_args["out_fp_base"],
        help=f"Path to output directory for lam files. Default:
        \"{default_args['out_fp_base']}\"",
    )
    parser.add_argument(
        "--exp_str",
        "-e",
        type=str,
        default=default_args["exp_str"],
        help=f"Experiment string. Default: \"{default_args['exp_str']}\"",
    )
    parser.add_argument(
        "--batch_size",
        "-b",
        type=int,
        default=default_args["batch_size"],
        help=f"Batch size. Default: {default_args['batch_size']}",
    )
    parser.add_argument(
        "--loss_lambda",
        "-l",
        type=float,
        default=default_args["loss_lambda"],
        help=f"Lambda for distribution loss. Default:
        {default_args['loss_lambda']}",
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=default_args["seed"],
        help=f"Random seed. Default: {default_args['seed']}",
    )

    parser.set_defaults(keep_existing_lams=True)
    parser.add_argument(
        "--keep_existing_lams",
        "-k",
        dest="keep_existing_lams",
        action="store_true",
        help="Keep existing lam files. Default: True",
    )
    parser.add_argument(
        "--no-keep_existing_lams",
        "-nk",
        dest="keep_existing_lams",
        action="store_false",
        help=(
            "Do not keep existing lam files. This will overwrite existing"
            " lam files. Default: False"
        ),
    )

    parser.set_defaults(use_distribution_loss=True)
    parser.add_argument(
        "--use_distribution_loss",
        "-d",
        dest="use_distribution_loss",
        action="store_true",
        help="Use distribution loss. Default: True",
    )
    parser.add_argument(
        "--no-use_distribution_loss",
        "-nd",
        dest="use_distribution_loss",
        action="store_false",
        help="Do not use distribution loss. Default: False",
    )

    args = parser.parse_args()
    logging.info("args are %s", args)

    main(
        raw_imgs_fp=args.raw_imgs_fp,
        model_fp=args.model_fp,
        out_fp_base=args.out_fp_base,
        exp_str=args.exp_str,
        loss_lambda=args.loss_lambda,
        batch_size=args.batch_size,
        use_distribution_loss=args.use_distribution_loss,
        seed=args.seed,
        keep_existing_lams=args.keep_existing_lams,
    )
