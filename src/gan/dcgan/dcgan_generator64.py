import torch
from torch import nn


class DCGANGenerator64(nn.Module):

    def __init__(self, out_channels, features=4):
        super().__init__()

        self.upsampling_block_64 = nn.Sequential(
            nn.ConvTranspose2d(features * 256, features * 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(features * 128),
            nn.ReLU(),
            nn.ConvTranspose2d(features * 128, features * 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(features * 64),
            nn.ReLU(),
            nn.ConvTranspose2d(features * 64, features * 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(features * 32),
            nn.ReLU(),
            nn.ConvTranspose2d(features * 32, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.upsampling_block_64(x)
