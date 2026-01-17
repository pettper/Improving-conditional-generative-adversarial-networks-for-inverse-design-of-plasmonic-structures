import time
import torch
from torch.nn import L1Loss
from src.utils import gradient_penalty, print_gan_losses, estimate_reconstruction_error, estimate_forward_error
from src.gan.base_gan_trainer import BaseGanTrainer
import os

# PARAMETERS
Z_DIM = 100
LAMBDA = 10
N_CRITIC = 5

# Fixed noise for validation
N_IMAGES = 16  # Number of validation images to generate
fixed_noise = torch.normal(0, 1, size=(N_IMAGES, Z_DIM))


class WGANTrainer(BaseGanTrainer):

    def __init__(self, critic, generator, critic_optimizer, generator_optimizer, training_loader, validation_loader,
                 device, forward_network=None, load_model_filename=None, save_model_filename=None,
                 write_to_tensorboard=False):

        """
        Trainer object for training a Wasserstein GAN with gradient penalty
        :param critic: critic network (nn.Module)
        :param generator: generator network (nn.Module)
        :param critic_optimizer: optimizer for the critic network
        :param generator_optimizer: optimizer for the generator network
        :param training_loader: DataLoader for the training set
        :param validation_loader: DataLoader for the validation set
        :param device: torch.device to run on
        :param load_model_filename: Filename to load model from
        :param save_model_filename: Filename to save model to
        """

        super().__init__(critic, generator, critic_optimizer, generator_optimizer, training_loader, validation_loader,
                         device, forward_network=forward_network, load_model_filename=load_model_filename,
                         save_model_filename=save_model_filename, write_to_tensorboard=write_to_tensorboard)

    def train_one_epoch(self):
        # Set models to training mode
        self.critic.train(True)
        self.generator.train(True)

        # Initialize loss calculations
        running_critic_loss = 0
        running_gen_loss = 0
        running_wasserstein_distance = 0
        it = 0

        start = time.perf_counter()
        for i, (images, labels) in enumerate(self.training_loader):
            # Send to device
            images, labels = images.to(self.device), labels.to(self.device)

            # Train the critic for N_CRITIC updates
            for t in range(N_CRITIC):
                critic_loss, wasserstein_distance = self.train_critic(images, labels)
                # Compute average loss on last critic update
                if t == N_CRITIC - 1:
                    running_critic_loss += critic_loss
                    running_wasserstein_distance += wasserstein_distance

            # Train the generator
            running_gen_loss += self.train_generator(labels)
            it += 1
        end = time.perf_counter()
        elapsed = end - start
        return running_critic_loss / it, running_gen_loss / it, elapsed, running_wasserstein_distance / it

    def train_model(self, epochs):

        epoch_times = torch.zeros(epochs)
        best_val_rce_mean = 1e100 # Start with something large

        # Train the model for 'epochs' number of epochs
        for epoch in range(epochs):
            
            # Train one epoch, collect losses and time
            critic_loss, gen_loss, elapsed, wasserstein_distance = self.train_one_epoch()
            epoch_times[epoch] = elapsed

            # Print losses and write images to tensorboard
            if epoch % 10 == 0:
                print_gan_losses(self.last_epoch + epoch + 1, (critic_loss, gen_loss), elapsed)

            if self.write_to_tensorboard:
                im_grids, images = self.get_image_grids()
                self.write_images_to_tensorboard(self.last_epoch + epoch, im_grids)

            # Store Wasserstein distance
            self.wasserstein_distance.append([self.last_epoch + epoch, wasserstein_distance])

            # Save images for plotting every 25th epoch
            # if epoch % 25 == 0 and self.save_image_dir:
            #    self.save_images(self.last_epoch + epoch, images)

            
            """
            The contents in the below if statement are very slow
            Tests show that for the full dataset ~3000 samples. It takes ~4.5 seconds to evaluate it.
            """
            # Monitor the reconstruction error every 50th epoch
            if epoch % 50 == 0:
                # On training data...
                train_rce_mean, train_rce_var, train_struct_rce_mean, train_struct_rce_var = estimate_reconstruction_error(self.generator, self.training_loader,
                                                                              self.device, metric=L1Loss())
                # On validation data...
                val_rce_mean, val_rce_var, val_struct_rce_mean, val_struct_rce_var = estimate_reconstruction_error(self.generator, self.validation_loader,
                                                                          self.device, metric=L1Loss())
                
                self.reconstruction_error.append([self.last_epoch + epoch, train_rce_mean, train_rce_var, val_rce_mean,
                                                  val_rce_var, train_struct_rce_mean, train_struct_rce_var, val_struct_rce_mean,
                                                  val_struct_rce_var])
                
                if self.write_images_to_tensorboard:
                    self.write_reconstruction_error_to_tensorboard(self.last_epoch + epoch, train_rce_mean, val_rce_mean, train_struct_rce_mean, val_struct_rce_mean)

                # If a forward network is provided, store the forward error
                if self.forward_network is not None:
                    train_forward_error, train_cnn_error = estimate_forward_error(self.generator, self.training_loader,
                                                                                  self.device, self.forward_network)
                    val_forward_error, val_cnn_error = estimate_forward_error(self.generator, self.validation_loader,
                                                                              self.device, self.forward_network)
                    self.forward_error_values.append([self.last_epoch + epoch,
                                                      train_forward_error,
                                                      train_cnn_error,
                                                      val_forward_error,
                                                      val_cnn_error])

            # Save model on last epoch and every 50th
            if self.save_model_filename:
                if (epoch + 1) == epochs:
                    self.save_checkpoint(suffix=f"_last_{epoch}")
                elif epoch % 50 == 0 and val_rce_mean < best_val_rce_mean:
                    self.save_checkpoint(suffix=f"_best_epoch_{epoch}")
                    best_val_rce_mean = val_rce_mean

        # Flush and close
        self.writer.flush()
        self.writer.close()

        return epoch_times

    def train_critic(self, real, labels):
        n_samples, c, h, w = real.shape

        # Sample from generator and interpolate
        z = torch.normal(0, 1, size=(n_samples, Z_DIM)).to(self.device)
        fake = self.generator(z, labels)
        eps = torch.rand(n_samples, 1, 1, 1).repeat(1, c, h, w).to(self.device)
        interpolated = eps * real + (1 - eps) * fake

        # Calculate critic loss
        wasserstein_distance = self.critic(real, labels) - self.critic(fake, labels)
        critic_loss = (-wasserstein_distance +
                       LAMBDA * gradient_penalty(self.critic, interpolated, labels).view((n_samples, 1)))
        critic_loss = critic_loss.mean()  # Computes mean over the batch

        # Update critic
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        return critic_loss.item(), wasserstein_distance.mean().item()

    def train_generator(self, labels):
        n_samples = labels.shape[0]

        # Calculate generator loss
        z = torch.normal(0, 1, size=(n_samples, Z_DIM)).to(self.device)
        generator_loss = -self.critic(self.generator(z, labels), labels).mean()

        # Update generator
        self.gen_optim.zero_grad()
        generator_loss.backward()
        self.gen_optim.step()

        return generator_loss.item()
