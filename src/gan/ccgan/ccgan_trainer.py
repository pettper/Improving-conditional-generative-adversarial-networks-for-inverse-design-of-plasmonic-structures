import torch
import time
import os
from torch.nn import L1Loss
from src.gan import BaseGanTrainer
from src.utils import print_gan_losses, estimate_reconstruction_error, estimate_forward_error
from src.gan.ccgan import (compute_soft_weight, get_fake_labels, get_fake_images, get_noise_labels,
                           compute_soft_vicinity_data, critic_loss)
from torchvision.utils import make_grid
import numpy as np
from matplotlib import pyplot as plt

# Parameters
THRESHOLD = 0.001
Z_DIM = 100
N_CRITIC = 5

# Fixed noise for validation
N_IMAGES = 16  # Number of validation images to generate
fixed_noise = torch.normal(0, 1, size=(N_IMAGES, Z_DIM))


class CcGANTrainer(BaseGanTrainer):

    def __init__(self, critic, generator, critic_optimizer, generator_optimizer, label_autoencoder, training_loader,
                 validation_loader, hyperparams, device, forward_network=None,
                 save_model_filename=None, load_model_filename=None, write_to_tensorboard=False):
        """
        Initializes a CcGANTrainer object (See CcGAN paper, supplementary materials for algorithm) with
        Wasserstein distance based loss.

        :param device:
        :param label_autoencoder: Pretrained autoencoder model to encode vector into a single dimension
        :param generator: The generator model
        :param critic: The critic model
        :param generator_optimizer: Generators optimizer
        :param critic_optimizer: Critics optimizer
        :param training_loader: Training data loader
        :param hyperparams: Dictionary with hyperparameters 'sigma', 'nu', and 'LAMBDA' (Key values)
        """

        super().__init__(critic, generator, critic_optimizer, generator_optimizer, training_loader, validation_loader,
                         device, forward_network=forward_network, load_model_filename=load_model_filename,
                         save_model_filename=save_model_filename, write_to_tensorboard=write_to_tensorboard)
        self.autoencoder = label_autoencoder
        self.SIGMA = hyperparams['sigma']
        self.NU = hyperparams['nu']
        self.LAMBDA = hyperparams['LAMBDA']
        # Autoencoder is only used for evaluation, set to eval mode
        self.autoencoder.eval()

    def train_model(self, epochs):
        # Train model for 'epochs' number of epochs
        for epoch in range(epochs):
            # Train one epoch, collect losses and time
            crit_loss, gen_loss, elapsed, wasserstein_distance = self.train_one_epoch()

            # Print losses and write images to tensorboard
            if epoch % 50 == 0:
                print_gan_losses(epoch + 1, (crit_loss, gen_loss), elapsed)
            im_grids, images = self.get_image_grids()
            self.write_images_to_tensorboard(epoch, im_grids)

            # Store Wasserstein distance
            self.wasserstein_distance.append([epoch, wasserstein_distance])

            # Save images for plotting every 250th epoch
            if epoch % 250 == 0 and self.save_image_dir:
                self.save_images(epoch, images)

            # Monitor the reconstruction error every 5th epoch
            if epoch % 100 == 0:
                # On training data...
                train_rce_mean, train_rce_var = estimate_reconstruction_error(self.generator, self.training_loader,
                                                                              self.device, metric=L1Loss())
                # On validation data...
                val_rce_mean, val_rce_var = estimate_reconstruction_error(self.generator, self.validation_loader,
                                                                          self.device, metric=L1Loss())
                self.reconstruction_error.append([epoch, train_rce_mean, train_rce_var, val_rce_mean, val_rce_var])
                self.write_reconstruction_error_to_tensorboard(epoch, val_rce_mean, val_rce_var)
                # If a forward network is provided, store the forward error
                if self.forward_network is not None:
                    train_forward_error = estimate_forward_error(self.generator, self.training_loader, self.device,
                                                                 self.forward_network)
                    val_forward_error = estimate_forward_error(self.generator, self.validation_loader, self.device,
                                                               self.forward_network)
                    self.forward_error_values.append([epoch, train_forward_error, val_forward_error])

            # Save model on last epoch and every 50th
            if ((epoch + 1) == epochs or epoch % 50 == 0) and self.save_model_filename:
                self.save_checkpoint()

            # Flush and close
        self.writer.flush()
        self.writer.close()

    def train_one_epoch(self):
        self.critic.train(True)
        self.generator.train(True)

        # Update discriminator
        disc_loss = 0.0
        wasserstein_distance = 0.0
        start = time.perf_counter()
        for t in range(N_CRITIC):
            iterator = iter(self.training_loader)
            images, labels = next(iterator)
            images, labels = images.to(self.device), labels.to(self.device)
            disc_loss, wasserstein_distance = self.update_critic(images, labels)

        # Update generator
        iterator = iter(self.training_loader)
        images, labels = next(iterator)
        images, labels = images.to(self.device), labels.to(self.device)
        gen_loss = self.update_generator(labels)

        end = time.perf_counter()
        elapsed = end - start
        return disc_loss, gen_loss, elapsed, wasserstein_distance

    def update_critic(self, images, labels):
        batch_size = images.shape[0]
        """
        plt.figure()
        index = 2
        with torch.no_grad():
            plt.plot(labels[index], label='original')
            plt.plot(self.autoencoder(labels)[index], label='original, decoded')
        """
        # Get noise labels
        noise_labels = self.autoencoder.decode(get_noise_labels(self.autoencoder.encode(labels),
                                                                self.device, batch_size))
        labels = self.autoencoder(labels)   # Use encoded decoded labels to compute soft vicinity data

        # Real image data
        real_idx, real_labels, real_soft_weights = compute_soft_vicinity_data(labels, noise_labels, self.autoencoder,
                                                                              self.device, batch_size, THRESHOLD,
                                                                              self.NU, self.SIGMA)
        real_images = images[real_idx]

        # Fake labels
        fake_labels = self.autoencoder.decode(
            get_fake_labels(self.autoencoder.encode(real_labels), self.device,self.NU, THRESHOLD))
        fake_soft_weights = compute_soft_weight(fake_labels, real_labels, self.NU)
        """
        with torch.no_grad():
            plt.plot(real_labels[index], label='real')
            plt.plot(fake_labels[index], label='fake')
        plt.legend()
        plt.show()
        """
        # Fake images
        fake_images = get_fake_images(self.generator, fake_labels, Z_DIM, self.device)

        # Compute loss
        loss, wasserstein_distance = critic_loss(self.critic, self.device, real_images, fake_images, real_labels, real_soft_weights,
                                                 fake_soft_weights, self.LAMBDA)

        # Step
        self.critic_optim.zero_grad()
        loss.backward()
        self.critic_optim.step()

        return loss, wasserstein_distance

    def update_generator(self, labels):
        batch_size = labels.shape[0]

        # Get noise labels
        noise_labels = self.autoencoder.decode(
            get_noise_labels(self.autoencoder.encode(labels), self.device, batch_size))

        # Get fake images
        fake_images = get_fake_images(self.generator, noise_labels, Z_DIM, self.device)

        # Compute loss
        loss = -self.critic(fake_images, noise_labels).mean()

        # Step
        self.gen_optim.zero_grad()
        loss.backward()
        self.gen_optim.step()

        return loss
