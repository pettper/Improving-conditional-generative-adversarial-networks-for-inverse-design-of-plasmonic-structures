import torch
from torch import nn

from src.gan.utils import LabelEmbeddingNetwork


class FullyConnectedGenerator(nn.Module):
    # Adds a more exotic way to input the labels to the generator

    def __init__(self, out_channels, target_channels=2, z_dim=100, y_dim=41, features=4, image_size=128, proj_dim=50,
                 embed=True, dropout_rate=0.5):
        super().__init__()

        self.z_dim = z_dim
        self.y_dim = y_dim
        self.proj_dim = proj_dim
        self.image_size = image_size
        self.embed = embed
        self.channels = out_channels
        self.target_channels = target_channels

        self.mlp = nn.Sequential(
            nn.Linear(in_features=proj_dim, out_features=64*features),
            nn.ReLU(),
            nn.Linear(in_features=64*features, out_features=64*features),
            nn.ReLU(),
            nn.Linear(in_features=64*features, out_features=64*features),
            nn.ReLU(),
            nn.Linear(in_features=64 * features, out_features=64 * features),
            nn.ReLU(),
            nn.Linear(in_features=64*features, out_features=out_channels*image_size*image_size),
            nn.Tanh(),
            nn.Unflatten(dim=1, unflattened_size=(out_channels, image_size, image_size))
        )

        # Label embedding network
        self.label_embedding = LabelEmbeddingNetwork(proj_dim, channels=target_channels, activation=nn.ReLU(), dropout_rate=dropout_rate)

        # Prepare for addition layer
        self.add_layer = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(in_features=target_channels * y_dim, out_features=target_channels * y_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(in_features=target_channels * y_dim, out_features=proj_dim),
            nn.ReLU()
        )

        self.noise_layer = nn.Sequential(
            nn.Linear(in_features=z_dim, out_features=proj_dim),
            nn.ReLU(),
        )

    def forward(self, z, y):
        # z is noise, z -> (B, z_dim)
        # y is some conditional data. Either y -> (B, N, y_dim) or (y -> (B, y_dim))

        # Transform input
        z = self.noise_layer(z)
        if self.embed:
            e = self.label_embedding(y)
        else:
            e = self.add_layer(y)

        # Addition + upsampling
        e = e + z
        x = self.mlp(e)
        return x

    def get_z_dim(self):
        return self.z_dim

    def get_y_dim(self):
        return self.y_dim

    def get_channels(self):
        return self.channels

    def get_target_channels(self):
        return self.target_channels
