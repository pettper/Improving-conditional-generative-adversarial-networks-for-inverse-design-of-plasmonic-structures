from src.utils.blocks import TransposedInvertedResidualBlock
import torch
from torch import nn


class GeneratorV2(nn.Module):

    def __init__(self, out_channels, z_dim=100, y_dim=41, features=4):
        super().__init__()

        self.z_dim = z_dim
        self.y_dim = y_dim

        self.input_layer = nn.Sequential(
            nn.Linear(in_features=z_dim, out_features=features * 128 * 4 * 4, bias=False),
            nn.Unflatten(1, (features * 128, 4, 4))
        )

        self.label_input_layer = nn.Sequential(
            nn.Linear(in_features=y_dim, out_features=features * 128 * 4 * 4, bias=False),
            nn.Unflatten(1, (features * 128, 4, 4))
        )

        self.upsampling_block = nn.Sequential(
            TransposedInvertedResidualBlock(features * 256, features * 128, expansion_ratio=1, kernel_size=4, stride=2,
                                            padding=1),
            TransposedInvertedResidualBlock(features * 128, features * 128, expansion_ratio=1, kernel_size=3, stride=1,
                                            padding=1),
            nn.ReLU(),
            TransposedInvertedResidualBlock(features * 128, features * 64, expansion_ratio=1, kernel_size=4, stride=2,
                                            padding=1),
            TransposedInvertedResidualBlock(features * 64, features * 64, expansion_ratio=1, kernel_size=3, stride=1,
                                            padding=1),
            nn.ReLU(),
            TransposedInvertedResidualBlock(features * 64, features * 32, expansion_ratio=1, kernel_size=4, stride=2,
                                            padding=1),
            TransposedInvertedResidualBlock(features * 32, features * 32, expansion_ratio=1, kernel_size=3, stride=1,
                                            padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(features * 32, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, z, y):
        # z is noise, y is some conditional data

        # Transform the tensor input to the
        z = self.input_layer(z)
        y = self.label_input_layer(y)

        # Concatenate
        x = torch.cat((z, y), dim=1)
        return self.upsampling_block(x)

    def get_z_dim(self):
        return self.z_dim

    def get_y_dim(self):
        return self.y_dim
