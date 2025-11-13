from torch import nn


# Class for an inverted residual block
class InvertedResidualBlock(nn.Module):

    def __init__(self, inp_channels, output_channels, expansion_ratio=6, kernel_size=3, stride=1, padding=1, bias=False,
                 normalization_layer=nn.BatchNorm2d, activation=nn.ReLU6()):
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
        self.hasResidualConnection = inp_channels == output_channels and stride == 1

        self.block = nn.Sequential(
            # 1x1 convolution
            nn.Conv2d(in_channels=inp_channels, out_channels=hidden_dim, kernel_size=1, stride=1, bias=bias),
            normalization_layer(hidden_dim, affine=True),
            activation,
            # depthwise convolution
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=kernel_size, stride=stride,
                      padding=padding, groups=hidden_dim, bias=bias),
            normalization_layer(hidden_dim, affine=True),
            activation,
            # 1x1 convolution
            nn.Conv2d(in_channels=hidden_dim, out_channels=output_channels, kernel_size=1, stride=1, bias=bias),
            normalization_layer(output_channels, affine=True)
        )

    def forward(self, x):
        y = self.block(x)
        # Residual connection, if possible
        if self.hasResidualConnection:
            y = x + y
        return y
