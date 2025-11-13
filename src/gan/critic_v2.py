import torch
from torch import nn
from src.utils.blocks import InvertedResidualBlock

# Parameters
# Leaky ReLU slope in dcgan critic
SLOPE = 0.2


class CriticV2(nn.Module):

    def __init__(self, inp_channels, label_size, features=4):
        super().__init__()

        # Makes use of the inverted residual block from the efficient net architecture
        self.downsampling_block = nn.Sequential(
            nn.Conv2d(inp_channels, features * 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(SLOPE),
            InvertedResidualBlock(features * 32, features * 64, expansion_ratio=1, kernel_size=4, stride=2,
                                  normalization_layer=nn.InstanceNorm2d, activation=nn.LeakyReLU(SLOPE)),
            InvertedResidualBlock(features * 64, features * 64, expansion_ratio=1, kernel_size=3, stride=1,
                                    normalization_layer=nn.InstanceNorm2d, activation=nn.LeakyReLU(SLOPE)),
            nn.LeakyReLU(SLOPE),
            InvertedResidualBlock(features * 64, features * 128, expansion_ratio=1, kernel_size=4, stride=2,
                                  normalization_layer=nn.InstanceNorm2d, activation=nn.LeakyReLU(SLOPE)),
            InvertedResidualBlock(features * 128, features * 128, expansion_ratio=1, kernel_size=3, stride=1,
                                    normalization_layer=nn.InstanceNorm2d, activation=nn.LeakyReLU(SLOPE)),
            nn.LeakyReLU(SLOPE),
            InvertedResidualBlock(features * 128, features * 256, expansion_ratio=1, kernel_size=4, stride=2,
                                  normalization_layer=nn.InstanceNorm2d, activation=nn.LeakyReLU(SLOPE)),
            InvertedResidualBlock(features * 256, features * 256, expansion_ratio=1, kernel_size=3, stride=1,
                                    normalization_layer=nn.InstanceNorm2d, activation=nn.LeakyReLU(SLOPE)),
            nn.LeakyReLU(SLOPE),
            nn.Conv2d(features * 256, 3*label_size, kernel_size=4, stride=2, padding=0, bias=False),
            nn.Flatten()
        )

        # Linear output layer
        self.output_layer = nn.Sequential(
            nn.Linear(in_features=4 * label_size, out_features=1)
        )

    def forward(self, x, y):
        # x is image data, y is some conditional data
        x = self.downsampling_block(x)
        # Concatenates y and x along the second dimension
        x = torch.cat((x, y), dim=1)
        return self.output_layer(x)
