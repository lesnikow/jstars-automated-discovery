# -*- coding: utf-8 -*-
"""plotters module
Define our plotter module that encapsulates attributes and methods related to
plotting methods that we use for our lunar anomalies project.
"""
import torch
import os
import pickle
import matplotlib.pyplot as plt
from pprint import pprint


class LossPlotterBasic:
    def __init__(self, experiment_directory, base_interval=int(2096 / 32)):
        super(LossPlotterBasic, self).__init__()
        self.experiment_directory = experiment_directory
        self.cuda = torch.torch.cuda.is_available()
        self.plot_and_save_losses_interval = base_interval
        self.base_interval = base_interval
        self.cuda_speedup_factor = 8
        if self.cuda:
            self.base_interval *= self.cuda_speedup_factor
            self.plot_and_save_losses_interval *= self.cuda_speedup_factor

    def plot_losses_statically(self, losses, epoch, batch_index):
        plt.figure(figsize=(15, 10))
        font_size = 16
        start_epoch_to_plot = 0
        plt.ylim(0, 1000)

        plt.plot(range(len(losses[start_epoch_to_plot:])), losses[start_epoch_to_plot:])
        plt.title("Train and Validation Losses")
        plt.xlabel("Steps", fontsize=font_size)
        plt.ylabel("Loss", fontsize=font_size)
        plt.legend(["VAE loss train", "VAE loss val"], fontsize=font_size - 4)

        plt.savefig(
            os.path.join(
                self.experiment_directory,
                "loss_graphs",
                ("a_train_val_losses_static" "graph_epoch{}_batch{}.png").format(
                    str(epoch), str(batch_index)
                ),
            )
        )
        plt.close()

    def plot_losses_dynamically(self, losses, epoch, batch_index):
        plt.figure(figsize=(15, 10))
        font_size = 16
        start_epoch_to_plot = 0
        plt.ylim(0, 1000)
        if not (epoch is 0 and batch_index is 0):
            try:
                plt.ylim(0, max(losses[-128:]))
                plt.axvline(x=max(0, losses.__len__() - 128), color="violet")
            except Exception as exception:
                pprint(
                    "Exception encountered when trying to dynamically set"
                    "y-axis limit of plot. Keeping default y-axis limit"
                    "setting."
                )
                print(exception)

        plt.plot(range(len(losses[start_epoch_to_plot:])), losses[start_epoch_to_plot:])
        plt.title("Train Losses")
        plt.xlabel("Steps", fontsize=font_size)
        plt.ylabel("Loss", fontsize=font_size)
        plt.legend(
            ["Dynamic Y-Axis-Scaling Input Window", "VAE loss train", "VAE loss val"],
            fontsize=font_size - 4,
        )
        plt.savefig(
            os.path.join(
                self.experiment_directory,
                "loss_graphs",
                ("b_train_val_losses_dynamic" "_graph_epoch{}_batch{}.png").format(
                    str(epoch), str(batch_index)
                ),
            )
        )
        plt.close()

    @staticmethod
    def write_out_train_losses_to_disk(model, exp_dir_fp):
        with open(
            os.path.join(
                exp_dir_fp,
                "loss_tables",
                "end_of_epoch_train_losses_{}.txt".format(model.epoch),
            ),
            "w",
        ) as f:
            epoch_counter = 0
            for loss in model.end_of_epochs_train_losses:
                f.write("epoch: {} \tloss: {:.2f}\n".format(epoch_counter, loss))
                epoch_counter += 1
