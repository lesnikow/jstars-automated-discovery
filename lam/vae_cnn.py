# -*- coding: utf-8 -*-
"""vae_cnn module
Define our vae_cnn module that encapsulates attributes and methods related to
the autoencoder model that we use for our lunar anomalies project.
"""
import os
import sys
import pickle
import torch

import numpy as np
from torch import nn
from torch import optim
from torch.autograd import Variable
from torchvision.utils import save_image

from data import LunarDataLoader
from plotters import LossPlotterBasic
from vae_losses import LossVAE


class VAE_CNN(nn.Module):
    def __init__(
        self,
        experiment_directory=None,
        image_channels=1,
        image_height_in_pixels=64,
        image_width_in_pixels=64,
        l_space_dim=None,
        batch_size=None,
        dim_reduction=128,
        verbose=True,
    ):
        # Setup
        nn.Module.__init__(self)
        self.experiment_directory = experiment_directory
        self.epoch = 0
        self.train_losses, val_losses = [], []
        self.end_of_epochs_train_losses, end_of_epochs_val_losses = [], []
        self.append_train_loss_to_train_losses_interval = 1
        self.base_interval = int(256)
        self.evaluate_reconstructions_interval = self.base_interval
        self.evaluate_generations_interval = self.base_interval
        self.display_vae_loss_interval = self.base_interval
        self.disable_cuda = False
        if not self.disable_cuda and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.l_space_dim = l_space_dim
        self.image_height_in_pixels = image_height_in_pixels
        self.image_width_in_pixels = image_width_in_pixels
        self.image_channels = image_channels
        self.dim_reduction = dim_reduction
        self.verbose = verbose

        # Encoder
        self.conv1 = nn.Conv2d(
            self.image_channels, 16, kernel_size=3, stride=1, padding=1, bias=True
        )
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn3 = nn.BatchNorm2d(64)

        # Note going from 64 to 16 channels here.
        self.conv4 = nn.Conv2d(
            64, self.dim_reduction, kernel_size=3, stride=2, padding=1, bias=True
        )
        self.bn4 = nn.BatchNorm2d(self.dim_reduction)

        # Latent vectors of means mu and variances sigma.
        self.num_features = (
            int(self.image_height_in_pixels / 4)
            * int(self.image_width_in_pixels / 4)
            * self.dim_reduction
        )
        self.fc1 = nn.Linear(self.num_features, self.l_space_dim)
        self.fc_bn1 = nn.BatchNorm1d(self.l_space_dim)
        self.fc21 = nn.Linear(self.l_space_dim, self.l_space_dim)
        self.fc22 = nn.Linear(self.l_space_dim, self.l_space_dim)

        # Decoder
        self.fc3 = nn.Linear(self.l_space_dim, self.l_space_dim)
        self.fc_bn3 = nn.BatchNorm1d(self.l_space_dim)
        self.fc4 = nn.Linear(self.l_space_dim, self.num_features)
        self.fc_bn4 = nn.BatchNorm1d(self.num_features)
        self.conv5 = nn.ConvTranspose2d(
            self.dim_reduction,
            64,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            bias=True,
        )
        self.bn5 = nn.BatchNorm2d(64)
        self.conv6 = nn.ConvTranspose2d(
            64, 32, kernel_size=3, stride=1, padding=1, bias=True
        )
        self.bn6 = nn.BatchNorm2d(32)
        self.conv7 = nn.ConvTranspose2d(
            32, 16, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True
        )
        self.bn7 = nn.BatchNorm2d(16)
        self.conv8 = nn.ConvTranspose2d(
            16, self.image_channels, kernel_size=3, stride=1, padding=1, bias=True
        )
        self.relu = nn.ReLU()

        if self.verbose:
            print(f"model is \n {self}")

    def encode(self, x):
        y = self.conv1(x)
        conv1 = self.relu(self.bn1(y))
        conv2 = self.relu(self.bn2(self.conv2(conv1)))
        conv3 = self.relu(self.bn3(self.conv3(conv2)))
        conv4 = self.relu(self.bn4(self.conv4(conv3))).view(-1, self.num_features)
        # TOTRY: conv5 = self.relu(self.bn)
        fc1 = self.relu(self.fc_bn1(self.fc1(conv4)))
        mu = self.fc21(fc1)
        sigma = self.fc22(fc1)
        return mu, sigma

    def reparametrize(self, mu, sigma):
        if self.training:
            # Transform log variance sigma to variance
            variance = sigma.mul(0.5).exp_()
            epsilon = Variable(variance.data.new(variance.size()).normal_())
            return epsilon.mul(variance).add_(mu)
        else:
            return mu

    def decode(self, z):
        fc3 = self.relu(self.fc_bn3(self.fc3(z)))
        fc4 = self.relu(self.fc_bn4(self.fc4(fc3)))
        fc4 = fc4.view(
            -1,
            self.dim_reduction,
            int(self.image_height_in_pixels / 4),
            int(self.image_width_in_pixels / 4),
        )
        conv5 = self.relu(self.bn5(self.conv5(fc4)))
        conv6 = self.relu(self.bn6(self.conv6(conv5)))
        conv7 = self.relu(self.bn7(self.conv7(conv6)))
        return self.conv8(conv7).view(
            -1,
            self.image_channels,
            self.image_height_in_pixels,
            self.image_width_in_pixels,
        )

    def forward(self, x):
        mu, sigma = self.encode(x)
        z = self.reparametrize(mu, sigma)
        return self.decode(z), mu, sigma

    def train_for_a_single_epoch(
        self,
        epoch,
        train_loader,
        optimizer,
        loss_to_use,
        plotter,
        device,
        batch_size=None,
    ):
        """Trains the model on a train dataset for a single epoch.
        Use this is e.g. our main script to train a model over multiple epochs
        of data.
        """
        display_losses = True
        make_generations_with_random_codes_while_training = False
        accumulated_train_loss = 0
        for batch_index, (data, _) in enumerate(train_loader):
            self.train()
            data = data.to(device)
            optimizer.zero_grad()
            reconstructed_batch, mu, sigma = self.forward(data)
            loss_computed_per_batch = loss_to_use.forward(
                reconstructed_batch, data, mu, sigma
            )
            loss_computed_per_batch.backward()
            accumulated_train_loss += loss_computed_per_batch.item()
            optimizer.step()

            if batch_index % self.append_train_loss_to_train_losses_interval == 0:
                loss_computed_average_per_sample = loss_computed_per_batch.item() / len(
                    data
                )
                self.train_losses.append(loss_computed_average_per_sample)

            if batch_index % plotter.plot_and_save_losses_interval == 0 and not (
                epoch is 0 and batch_index is 0
            ):
                plotter.plot_losses_statically(self.train_losses, epoch, batch_index)
                plotter.plot_losses_dynamically(self.train_losses, epoch, batch_index)

            if batch_index % self.evaluate_reconstructions_interval == 0:
                with torch.no_grad():
                    self.eval()
                    n = min(data.size(0), 8)
                    comparison = torch.cat(
                        [
                            data[:n],
                            reconstructed_batch.view(
                                -1,
                                self.image_channels,
                                self.image_height_in_pixels,
                                self.image_width_in_pixels,
                            )[:n],
                        ]
                    )
                    save_image(
                        comparison.cpu(),
                        os.path.join(
                            self.experiment_directory,
                            "reconstructions_and_generated_samples",
                            "reconstructions",
                            "reconstructions_epoch{}"
                            "_batch{}.png".format(str(epoch), str(batch_index)),
                        ),
                        nrow=n,
                    )

            if (
                display_losses
                and batch_index % self.display_vae_loss_interval == 0
                and not (epoch is 0 and batch_index is 0)
            ):
                loss_to_use.print_reconstruction_and_divergence_losses()
                loss_computed_averaged_per_sample = (
                    loss_computed_per_batch.item() / len(data)
                )
                print(
                    "Train Epoch: {0} [{1}/{2} ({3:.0f}%)]\tVAE Loss: {4:.6f}".format(
                        epoch,
                        batch_index * len(data),
                        len(train_loader.dataset),
                        100.0 * batch_index * len(data) / len(train_loader.dataset),
                        loss_computed_average_per_sample,
                    )
                )

            if (
                make_generations_with_random_codes_while_training
                and batch_index % self.evaluate_generations_interval == 0
            ):
                with torch.no_grad():
                    self.eval()
                    sample_latent_vectors = torch.randn(
                        batch_size, self.l_space_dim
                    ).to(device)
                    sample_images = self.decode(sample_latent_vectors).cpu()
                    save_image(
                        sample_images,
                        os.path.join(
                            self.experiment_directory,
                            "reconstructions_and_generated_samples",
                            "generated_samples",
                            "samples_epoch_{}"
                            "_batch{}_.png".format(str(epoch), str(batch_index)),
                        ),
                    )

        self.end_of_epochs_train_losses.append(
            accumulated_train_loss / len(train_loader.dataset)
        )
        self.epoch += 1
        print(
            "=====> Epoch: {} Train average VAE loss: {:.4f}".format(
                epoch, accumulated_train_loss / len(train_loader.dataset)
            )
        )

    def validate_on_a_val_or_test_set(
        self,
        val_or_test_loader: LunarDataLoader = None,
        device: torch.device = None,
        batch_size: int = None,
        use_reconstruction_loss: bool = True,
        use_distribution_loss: bool = True,
        reconstruction_loss_metric: torch.nn.modules.loss._Loss = None,
        loss_lambda: float = None,
        batch_break_value: int = None,
        verbose: bool = True,
        vv: bool = False,
    ) -> ([float], [float]):
        """Validates the model on a validation or test dataset.
        Use this to generate e.g. precision-recall curves to evaluate our trained models
        on a validation or test dataset.

        Args:
            val_or_test_loader: The torch DataLoader to use.
            device: The torch.device to use.
            batch_size: The batch size to use.
            use_reconstruction_loss: Whether to use the reconstruction
                componenent of the vae loss.
            use_distribution_loss: Whether to use the distribution componenent
                of the vae loss.
            reconstruction_loss_metric: The distance metric to use for the
                reconstruction loss. Examples include nn.MSELoss() or
                nn.L1Loss(). These generally subclass from the torch.nn._Loss()
                parent class.
            loss_lambda: The lambda factor between reconstruction and
                distribution losses. The anomaly score is computed as
                reconstruction_loss + lambda * distribution loss.
            batch_break_value: The batch value to break our validation method
                on. Useful to validate or test on a small part of the val or test
                set.
            verbose: Whether to print information.
            vv: Whether to print more information.

        Returns:
            (y_labels, y_scores) A tuple of labels and the model's scores.
        """
        self.eval()
        y_labels, y_scores = [], []
        with torch.no_grad():
            for batch_index, (original_batch, label_batch) in enumerate(
                val_or_test_loader
            ):
                if batch_index >= batch_break_value:
                    break
                y_labels.extend(label_batch.tolist())
                original_batch = original_batch.to(device)
                reconstructed_batch, mu, sigma = self.forward(original_batch)

                reconstruction_loss_batch = reconstruction_loss_metric(
                    reduction="none"
                )(reconstructed_batch, original_batch)
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
                y_scores.extend(vae_anomaly_score_batch.tolist())

                if verbose:
                    print(
                        f"validating our model on batch_index {batch_index} "
                        f"with label_batch {label_batch}"
                    )
                if vv:
                    print(f"vae_anomaly_score_batch is {vae_anomaly_score_batch}")
                    print(
                        f"vae_anomaly_score_batch.shape is {vae_anomaly_score_batch.shape}"
                    )

        if vv:
            print(f"y_labels, y_scores are {y_labels[:4]}..., {y_scores[:4]}... .")
            print(f"y_labels, y_scores have lengths {len(y_labels)}, {len(y_scores)}")
        return y_labels, y_scores
