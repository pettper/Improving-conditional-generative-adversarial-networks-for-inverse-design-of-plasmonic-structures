from torch import nn
from src.utils.blocks import SqueezeExcitationBlock


# Class for an inverted residual block with transposed convolution
class SETransposedInvertedResidualBlock(nn.Module):

    def __init__(self, inp_channels, output_channels, expansion_ratio=6, reduction_ratio=4, kernel_size=4, stride=2,
                 padding=1, bias=False, normalization_layer=nn.BatchNorm2d, activation=nn.ReLU()):
        """
        Initialization method for an inverted residual block.

        :param inp_channels: The number of input channels to the block
        :param output_channels: The number of output channels to the block
        :param expansion_ratio: The expansion ratio of the hidden dimension
        :param kernel_size: The kernel size of the 3x3 convolution
        :param stride: The stride of the 3x3 convolution
        :param padding: The padding of the 3x3 convolution
        :param bias: Use bias term or not (all layers)
        :param normalization_layer: Use layer normalization after convolutional layer (Default is batch normalization).
        :param activation: Activation function for layers with activation (Default ReLu6)
        """
        super().__init__()

        hidden_dim = expansion_ratio * inp_channels
        reduced_dim = max(inp_channels // reduction_ratio, 1)
        self.hasResidualConnection = stride == 1
        self.channelsIsEqual = inp_channels == output_channels

        # Use a normal convolutional block if there is no stride
        if stride == 1:
            self.conv_block = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride,
                                        padding=padding, groups=hidden_dim, bias=bias)
        else:
            self.conv_block = nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride,
                                                 padding=padding, output_padding=stride-1, groups=hidden_dim, bias=bias)
        self.block = nn.Sequential(
            # 1x1 convolution
            nn.Conv2d(in_channels=inp_channels, out_channels=hidden_dim, kernel_size=1, stride=1, bias=bias),
            normalization_layer(hidden_dim, affine=True),
            activation,
            # depthwise convolution
            self.conv_block,
            normalization_layer(hidden_dim, affine=True),
            activation,
            # Squeeze excitation + 1x1 convolution
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
