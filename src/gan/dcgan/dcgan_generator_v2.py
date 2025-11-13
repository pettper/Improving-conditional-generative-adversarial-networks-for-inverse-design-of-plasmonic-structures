import torch
from torch import nn
from src.gan.dcgan import DCGANGenerator64, DCGANGenerator128


class DCGANGeneratorV2(nn.Module):

    def __init__(self, out_channels, z_dim=100, y_dim=80, features=4, image_size=128):
        super().__init__()

        self.z_dim = z_dim
        self.y_dim = y_dim
        self.image_size = image_size

        self.transform_block = nn.Sequential(
            nn.Linear(in_features=z_dim, out_features=features * 256 * 4 * 4),
            nn.ReLU(),
            nn.Unflatten(1, (features * 256, 4, 4))
        )

        self.noise_layer = nn.Sequential(
            nn.Linear(in_features=z_dim, out_features=z_dim),
            nn.ReLU(),
        )

        self.label_input_layer = nn.Sequential(
            nn.Linear(in_features=y_dim, out_features=z_dim),
            nn.ReLU()
        )

        self.upsampling_block_64 = DCGANGenerator64(out_channels)
        self.upsampling_block_128 = DCGANGenerator128(out_channels)

    def forward(self, z, y):
        # z is noise, y is some conditional data. y -> (B, 2, y_dim), z -> (B, z_dim)

        # Transform the tensor input to image channels
        z = self.noise_layer(z)
        y1 = self.label_input_layer(y[:, 0, :])
        y2 = self.label_input_layer(y[:, 1, :])
        y3 = self.label_input_layer(y[:, 2, :])
        y4 = self.label_input_layer(y[:, 3, :])

        # Addition + Naive label input
        x = z + y1 + y2 + y3 + y4
        x = self.transform_block(x)

        if self.image_size == 64:
            return self.upsampling_block_64(x)
        elif self.image_size == 128:
            return self.upsampling_block_128(x)
        else:
            raise Exception(f"Image size {self.image_size} is not supported")

    def get_z_dim(self):
        return self.z_dim

    def get_y_dim(self):
        return self.y_dim
