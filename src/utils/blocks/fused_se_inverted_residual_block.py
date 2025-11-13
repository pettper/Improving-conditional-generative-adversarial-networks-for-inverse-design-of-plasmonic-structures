from torch import nn
from src.utils.blocks import SqueezeExcitationBlock


class FusedSEInvertedResidualBlock(nn.Module):

    def __init__(self, inp_channels, output_channels, expansion_ratio=6, kernel_size=3, stride=1, padding=1,
                 reduction_ratio=4, activation=nn.ReLU(), normalization_layer=nn.BatchNorm2d, bias=False):
        super().__init__()

        hidden_dim = expansion_ratio * inp_channels
        reduced_dim = max(inp_channels // reduction_ratio, 1)
        self.hasResidualConnection = stride == 1
        self.channelsIsEqual = inp_channels == output_channels

        self.block = nn.Sequential(
            # Standard convolution
            nn.Conv2d(in_channels=inp_channels, out_channels=hidden_dim, kernel_size=kernel_size, stride=stride,
                      padding=padding, bias=bias),
            normalization_layer(hidden_dim, affine=True),
            activation,
            # Squeeze excitation block + 1x1 convolution
            SqueezeExcitationBlock(hidden_dim, reduced_dim),
            nn.Conv2d(in_channels=hidden_dim, out_channels=output_channels, kernel_size=1, stride=1, bias=bias),
            normalization_layer(output_channels, affine=True)
        )

        # 1x1 convolution to match input channels and output channels
        self.conv = nn.Conv2d(in_channels=inp_channels, out_channels=output_channels, kernel_size=1, stride=1,
                              padding=0, bias=bias)

    def forward(self, x):
        y = self.block(x)
        # Residual connection, if possible
        if self.hasResidualConnection:
            if self.channelsIsEqual:
                return x + y
            else:
                return self.conv(x) + y
        else:
            return y
