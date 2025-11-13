import numpy as np
import time
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from torch.nn import L1Loss
from src.utils import print_gan_losses, estimate_reconstruction_error

# PARAMETERS
Z_DIM = 100

# Fixed noise for validation
N_IMAGES = 32  # Number of validation images to generate
fixed_noise = torch.normal(0, 1, size=(N_IMAGES, Z_DIM))


class GANTrainer:

    def __init__(self, discriminator, generator, discriminator_optimizer, generator_optimizer, training_loader,
                 validation_loader, device, n_critic=1, load_model_filename=None, save_model_filename=None):

        """
        Trainer object for training a regular GAN-model
        :param discriminator: Discriminator model (nn.Module)
        :param generator: Generator model (nn.Module)
        :param discriminator_optimizer: Discriminator optimizer
        :param generator_optimizer: Generator optimizer
        :param training_loader: Dataloader for the training set
        :param validation_loader: Dataloader for the validations set
        :param device: torch.device to run on
        :param n_critic: Critic iterations per generator iteration
        :param load_model_filename: Filename to load from
        :param save_model_filename: Filename to save to
        """

        self.discriminator = discriminator
        self.generator = generator
        self.gen_optim = generator_optimizer
        self.disc_optim = discriminator_optimizer
        self.training_loader = training_loader
        self.validation_loader = validation_loader
        self.n_critic = n_critic
        self.writer = SummaryWriter()
        self.device = device

        if load_model_filename:
            self.load_checkpoint(load_model_filename)

        self.save_model_filename = save_model_filename
        self.reconstruction_error = []

    def train_one_epoch(self):
        # Set models to training mode
        self.discriminator.train(True)
        self.generator.train(True)

        # Initialize loss calculations
        running_disc_loss = 0
        running_gen_loss = 0
        it = 0

        start = time.perf_counter()
        for i, (real, labels) in enumerate(self.training_loader):
            real, labels = real.to(self.device), labels.to(self.device)
            n_samples, c, h, w = real.shape

            # Sample from generator
            z = torch.normal(0, 1, size=(n_samples, Z_DIM)).to(self.device)
            fake = self.generator(z, labels)

            # Train the critic for N_CRITIC updates
            for t in range(self.n_critic):
                running_disc_loss += self.train_discriminator(real, fake, labels)

            # Train the generator
            running_gen_loss += self.train_generator(fake, labels)
            it += 1
        end = time.perf_counter()
        elapsed = end - start

        return running_disc_loss / it, running_gen_loss / it, elapsed

    def train_model(self, epochs):
        # Train the model for 'epochs' number of epochs
        for epoch in range(epochs):
            # Train one epoch and collect losses
            disc_loss, gen_loss, elapsed = self.train_one_epoch()

            # Print losses and write images to tensorboard
            print_gan_losses(epoch + 1, (disc_loss, gen_loss), elapsed)
            self.write_images_to_tensorboard(epoch)

            # Monitor the reconstruction error every 5th epoch
            if epoch % 5 == 0:
                # On training data...
                train_rce_mean, train_rce_var = estimate_reconstruction_error(self.generator, self.training_loader,
                                                                              self.device, metric=L1Loss())
                # On validation data...
                val_rce_mean, val_rce_var = estimate_reconstruction_error(self.generator, self.validation_loader,
                                                                          self.device, metric=L1Loss())
                self.reconstruction_error.append([epoch, train_rce_mean, train_rce_var, val_rce_mean, val_rce_var])
                self.write_reconstruction_error_to_tensorboard(epoch, val_rce_mean, val_rce_var)

            # Save model on last epoch
            if ((epoch + 1) == epochs or epoch % 100 == 0) and self.save_model_filename:
                self.save_checkpoint()

        # Flush and close
        self.writer.flush()
        self.writer.close()

    def train_discriminator(self, real, fake, labels):

        # Calculate discriminator loss
        real_loss = torch.log(self.discriminator(real, labels)).mean()
        fake_loss = torch.log(1 - self.discriminator(fake, labels)).mean()
        disc_loss = -(real_loss + fake_loss)

        # Update discriminator
        self.disc_optim.zero_grad()
        disc_loss.backward(retain_graph=True)
        self.disc_optim.step()

        return disc_loss

    def train_generator(self, fake, labels):

        # Calculate generator loss
        generator_loss = -torch.log(self.discriminator(fake, labels)).mean()

        # Update generator
        self.gen_optim.zero_grad()
        generator_loss.backward()
        self.gen_optim.step()

        return generator_loss

    def write_images_to_tensorboard(self, epoch, n_images=N_IMAGES):
        # Evaluation mode
        self.generator.eval()
        with torch.no_grad():
            real, labels = next(iter(self.validation_loader))
            real, labels = real.to(self.device), labels.to(self.device)

            if real.shape[0] < n_images:
                n_images = real.shape[0]

            fake = self.generator(fixed_noise[:n_images].to(self.device), labels[:n_images])

            # Stores up to n_images
            img_grid_real = make_grid(real[:n_images], normalize=True)
            img_grid_fake = make_grid(fake[:n_images], normalize=True)

            self.writer.add_image("Image/Real", img_grid_real, epoch)
            self.writer.add_image("Image/Fake", img_grid_fake, epoch)

    def write_reconstruction_error_to_tensorboard(self, epoch, rce_mean, rce_var):
        self.writer.add_scalar("Reconstruction_error/Mean", rce_mean, epoch)
        self.writer.add_scalar("Reconstruction_error/Variance", rce_var, epoch)

    def save_checkpoint(self):
        state = {
            'generator_state_dict': self.generator.state_dict(),
            'generator_optimizer_state_dict': self.gen_optim.state_dict(),
            'critic_state_dict': self.discriminator.state_dict(),
            'critic_optimizer_state_dict': self.disc_optim.state_dict(),
            'reconstruction_error': np.array(self.reconstruction_error)
        }
        torch.save(state, self.save_model_filename)
        print(f"Saved checkpoint to, \"{self.save_model_filename}\"")

    def load_checkpoint(self, filename):
        checkpoint = torch.load(filename)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.gen_optim.load_state_dict(checkpoint['generator_optimizer_state_dict'])
        self.discriminator.load_state_dict(checkpoint['critic_state_dict'])
        self.disc_optim.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        print(f"Loaded model, \"{filename}\"")
