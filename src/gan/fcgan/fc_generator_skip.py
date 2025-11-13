import torch
from torch import nn

from src.gan.utils import LabelProjection


class FullyConnectedGeneratorSkip(nn.Module):
    # Adds a more exotic way to input the labels to the generator

    def __init__(self, out_channels, z_dim=100, y_dim=41, features=4, image_size=128, proj_dim=50):
        super().__init__()

        self.z_dim = z_dim
        self.y_dim = y_dim
        self.proj_dim = proj_dim
        self.image_size = image_size
        hidden_size = 128 * features
        # Input layer
        self.input = nn.Sequential(
            nn.Linear(in_features=proj_dim, out_features=hidden_size),
            nn.ReLU()
        )
        # 2 pcs multilayer perceptron hidden layers
        self.mlp = nn.ModuleList([nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.ReLU()) for _ in range(3)]
        )
        # 3 pcs skip connection layers
        self.skip = nn.ModuleList([nn.Sequential(
            nn.Linear(in_features=proj_dim, out_features=hidden_size//2),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size//2, out_features=hidden_size)) for _ in range(4)]
        )
        # Output layer
        self.output = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=out_channels*image_size*image_size),
            nn.Tanh(),
            nn.Unflatten(dim=1, unflattened_size=(out_channels, image_size, image_size))
        )
        # Label projection
        self.label_projection = LabelProjection(proj_dim, activation=nn.ReLU())
        self.noise_layer = nn.Sequential(
            nn.Linear(in_features=z_dim, out_features=proj_dim),
            nn.ReLU(),
        )

    def forward(self, z, y):
        # z is noise, z -> (B, z_dim)
        # y is some conditional data. Either y -> (B, N, y_dim) or (y -> (B, y_dim))

        # Transform input
        z = self.noise_layer(z)
        e = self.label_projection.embed(y).view(-1, self.proj_dim)
        e = e + z
        # Upsampling with skip connections
        x = self.input(e) + self.skip[0](e)
        x = self.mlp[0](x) + self.skip[1](e)
        x = self.mlp[1](x) + self.skip[2](e)
        x = self.mlp[2](x) + self.skip[3](e)
        return self.output(x)

    def get_z_dim(self):
        return self.z_dim

    def get_y_dim(self):
        return self.y_dim
