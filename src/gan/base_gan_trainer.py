import torch
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
from torchvision.utils import make_grid

# Set seed
torch.manual_seed(23)

# Parameters
Z_DIM = 100
N_IMAGES = 16  # Number of validation images to generate
# Fixed noise for validation
fixed_noise = torch.normal(0, 1, size=(N_IMAGES, Z_DIM))


class BaseGanTrainer:

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

        self.critic = critic
        self.generator = generator
        self.gen_optim = generator_optimizer
        self.critic_optim = critic_optimizer
        self.training_loader = training_loader
        self.validation_loader = validation_loader
        self.writer = SummaryWriter()
        self.reconstruction_error = []
        self.device = device
        self.write_to_tensorboard = write_to_tensorboard
        self.forward_network = forward_network
        self.forward_error_values = []
        self.wasserstein_distance = []
        self.last_epoch = 0

        if load_model_filename:
            self.load_checkpoint(load_model_filename)
            self.last_epoch = self.wasserstein_distance[-1][0]

        self.save_model_filename = save_model_filename
        self.save_image_dir = None
        if self.save_model_filename:
            index_of_last_slash = save_model_filename.rfind('/')
            if index_of_last_slash == -1:
                self.save_image_dir = save_model_filename.split('.')[0]
            else:
                self.save_image_dir = (save_model_filename[:index_of_last_slash] +
                                       save_model_filename[index_of_last_slash:].split('.')[0])

        # Fixed samples for saving images
        if training_loader is not None:
            self.train_samples = next(iter(training_loader))
        if validation_loader is not None:
            self.validation_samples = next(iter(validation_loader))

    def get_image_grids(self, n_images=N_IMAGES):
        # Evaluation mode
        self.generator.eval()
        with torch.no_grad():
            train_real, train_labels = self.train_samples
            val_real, val_labels = self.validation_samples
            train_real, train_labels = train_real.to(self.device), train_labels.to(self.device)
            val_real, val_labels = val_real.to(self.device), val_labels.to(self.device)

            if val_real.shape[0] < n_images:
                n_images = val_real.shape[0]

            train_fake = self.generator(fixed_noise[:n_images].to(self.device), train_labels[:n_images])
            val_fake = self.generator(fixed_noise[:n_images].to(self.device), val_labels[:n_images])

            # Stores up to n_images
            train_img_grid_real = make_grid(train_real[:n_images, 0], normalize=True)
            train_img_grid_fake = make_grid(train_fake[:n_images, 0], normalize=True)
            val_img_grid_real = make_grid(val_real[:n_images, 0], normalize=True)
            val_img_grid_fake = make_grid(val_fake[:n_images, 0], normalize=True)
            im_grids = (train_img_grid_real, train_img_grid_fake, val_img_grid_real, val_img_grid_fake)
            images = (train_real[:n_images, ], train_fake[:n_images, ], val_real[:n_images, ], val_fake[:n_images, ])
        return im_grids, images

    def write_images_to_tensorboard(self, epoch, im_grids):
        train_img_grid_real, train_img_grid_fake, val_img_grid_real, val_img_grid_fake = im_grids
        self.writer.add_image("Training_Image/Real", train_img_grid_real, epoch)
        self.writer.add_image("Training_Image/Fake", train_img_grid_fake, epoch)
        self.writer.add_image("Validation_Image/Real", val_img_grid_real, epoch)
        self.writer.add_image("Validation_Image/Fake", val_img_grid_fake, epoch)

    def write_reconstruction_error_to_tensorboard(self, epoch, rce_mean, rce_var):
        if self.write_to_tensorboard:
            self.writer.add_scalar("Reconstruction_error/Mean", rce_mean, epoch)
            self.writer.add_scalar("Reconstruction_error/Variance", rce_var, epoch)

    def save_checkpoint(self):
        state = {
            'generator_state_dict': self.generator.state_dict(),
            'generator_optimizer_state_dict': self.gen_optim.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_optimizer_state_dict': self.critic_optim.state_dict(),
            'reconstruction_error': np.array(self.reconstruction_error),
            'forward_error': np.array(self.forward_error_values),
            'wasserstein_distance': self.wasserstein_distance
        }
        torch.save(state, self.save_model_filename)
        print(f"Saved checkpoint to, \"{self.save_model_filename}\"")

    def load_checkpoint(self, filename):
        checkpoint = torch.load(filename)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.gen_optim.load_state_dict(checkpoint['generator_optimizer_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_optim.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.reconstruction_error = checkpoint['reconstruction_error'].tolist()
        self.forward_error_values = checkpoint['forward_error'].tolist()
        self.wasserstein_distance = checkpoint['wasserstein_distance']
        print(f"Loaded model, \"{filename}\"")

    def save_images(self, epoch, images):
        train_real, train_fake, val_real, val_fake = images
        data = torch.stack((train_real, train_fake, val_real, val_fake)).to(torch.device('cpu')).numpy()

        # Make sure there is an available directory
        if not os.path.exists(self.save_image_dir):
            os.mkdir(self.save_image_dir)

        # Save to an .npy file
        np.save(str(self.save_image_dir) + f"/images_epoch_{epoch}", data)