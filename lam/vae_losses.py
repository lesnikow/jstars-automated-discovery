# -*- coding: utf-8 -*-
"""vae_losses module
Define our vae_losses module that encapsulates attributes and methods related to
the machine learning losses that we use for our lunar anomalies project.
"""
import torch
import plotters
import vae_cnn
import os
import sys
import pickle

from torch import nn
from torch import optim
from torch.autograd import Variable
from torchvision.utils import save_image


class LossVAE(nn.Module):
    def __init__(
        self,
        l1_or_l2_loss=None,
        loss_lambda=None,
        include_kld_loss=None,
        vae_loss_verbose=False,
        batch_size=None,
    ):
        super(LossVAE, self).__init__()
        self.l1_or_l2_loss = l1_or_l2_loss
        self.loss_lambda = loss_lambda
        self.include_kld_loss = include_kld_loss
        self.vae_loss_verbose = vae_loss_verbose
        self.loss_reconstruction_last = None
        self.loss_kld_last = None
        self.loss_reconstruction_array = []
        self.loss_kld_array = []
        self.batch_size = batch_size
        if l1_or_l2_loss == "l2":
            self.loss_reconstruction = nn.MSELoss(reduction="sum")
        elif l1_or_l2_loss == "l1":
            self.loss_reconstruction = nn.L1Loss(reduction="sum")
        else:
            raise ValueError("Please select l1_or_l2_loss out of 'l1' or 'l2'")

    def forward(self, reconstructed_sample, data, mu, sigma, sum_kld_loss=True):
        self.loss_reconstruction_last = self.loss_reconstruction(
            reconstructed_sample, data
        )
        if sum_kld_loss:
            self.loss_kld_last = -0.5 * torch.sum(
                1 - torch.pow(mu, 2) + sigma - torch.exp(sigma)
            )
        else:
            self.loss_kld_last = -0.5 * (
                torch.ones(len(mu)) - torch.pow(mu, 2) + sigma - torch.exp(sigma)
            )

        self.loss_reconstruction_array.append(self.loss_reconstruction_last)
        self.loss_kld_array.append(self.loss_kld_last)

        if self.vae_loss_verbose:
            self.print_reconstruction_and_divergence_losses()

        if self.include_kld_loss:
            return self.loss_reconstruction_last + self.loss_lambda * self.loss_kld_last
        else:
            return self.loss_reconstruction

    def print_reconstruction_and_divergence_losses(self):
        print(
            "Reconstruction, divergence losses, averaged per sample,"
            "loss_lambda are:\t\t\t{0:.3f}\t\t\t{1:.3f}\t\t\t{2:.2f}".format(
                self.loss_reconstruction_last / self.batch_size,
                self.loss_kld_last / self.batch_size,
                self.loss_lambda,
            )
        )
