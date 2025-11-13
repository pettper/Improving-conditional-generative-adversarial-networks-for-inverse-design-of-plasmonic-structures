from torch import nn
from src.utils.blocks import SEInvertedResidualBlock
from src.utils import conv_output_size

# Parameters
R_EXP = 6  # Expansion factor


class EfficientNet(nn.Module):

    def __init__(self, inp_channels, normalization_layer=nn.BatchNorm2d, image_size=64, activation=nn.ReLU()):
        super().__init__()
        self.image_size = image_size

        # Convolutional part of EfficientNet B0
        self.efficient_net_b0 = nn.Sequential(
            nn.Conv2d(inp_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            # Sequence 1
            SEInvertedResidualBlock(32, 16, expansion_ratio=1, kernel_size=3, stride=1,
                                    normalization_layer=normalization_layer, activation=activation),
            SEInvertedResidualBlock(16, 24, expansion_ratio=R_EXP, kernel_size=3, stride=1,
                                    normalization_layer=normalization_layer, activation=activation),
            SEInvertedResidualBlock(24, 24, expansion_ratio=R_EXP, kernel_size=3, stride=2,
                                    normalization_layer=normalization_layer, activation=activation),
            # Sequence 2
            SEInvertedResidualBlock(24, 40, expansion_ratio=R_EXP, kernel_size=5, stride=1, padding=2,
                                    normalization_layer=normalization_layer, activation=activation),
            SEInvertedResidualBlock(40, 40, expansion_ratio=R_EXP, kernel_size=5, stride=2, padding=2,
                                    normalization_layer=normalization_layer, activation=activation),
            # Sequence 3
            SEInvertedResidualBlock(40, 80, expansion_ratio=R_EXP, kernel_size=3, stride=1,
                                    normalization_layer=normalization_layer, activation=activation),
            SEInvertedResidualBlock(80, 80, expansion_ratio=R_EXP, kernel_size=3, stride=1,
                                    normalization_layer=normalization_layer, activation=activation),
            SEInvertedResidualBlock(80, 80, expansion_ratio=R_EXP, kernel_size=3, stride=2,
                                    normalization_layer=normalization_layer, activation=activation),
            # Sequence 4
            SEInvertedResidualBlock(80, 112, expansion_ratio=R_EXP, kernel_size=5, stride=1, padding=2,
                                    normalization_layer=normalization_layer, activation=activation),
            SEInvertedResidualBlock(112, 112, expansion_ratio=R_EXP, kernel_size=5, stride=1, padding=2,
                                    normalization_layer=normalization_layer, activation=activation),
            SEInvertedResidualBlock(112, 112, expansion_ratio=R_EXP, kernel_size=5, stride=1, padding=2,
                                    normalization_layer=normalization_layer, activation=activation),
            # Sequence 5
            SEInvertedResidualBlock(112, 192, expansion_ratio=R_EXP, kernel_size=5, stride=1, padding=2,
                                    normalization_layer=normalization_layer, activation=activation),
            SEInvertedResidualBlock(192, 192, expansion_ratio=R_EXP, kernel_size=5, stride=1, padding=2,
                                    normalization_layer=normalization_layer, activation=activation),
            SEInvertedResidualBlock(192, 192, expansion_ratio=R_EXP, kernel_size=5, stride=1, padding=2,
                                    normalization_layer=normalization_layer, activation=activation),
            SEInvertedResidualBlock(192, 192, expansion_ratio=R_EXP, kernel_size=5, stride=2, padding=2,
                                    normalization_layer=normalization_layer, activation=activation),
            # Sequence 6
            SEInvertedResidualBlock(192, 320, expansion_ratio=R_EXP, kernel_size=3, stride=1,
                                    normalization_layer=normalization_layer, activation=activation),
            # 1x1 conv and pooling
            nn.Conv2d(320, 1280, kernel_size=1),
            nn.AvgPool2d(kernel_size=self.get_avg_pool_size())
        )

    def forward(self, x):
        return self.efficient_net_b0(x)

    def get_avg_pool_size(self):
        s = conv_output_size(self.image_size, 3, 1, 2)
        s = conv_output_size(s, 3, 1, 2)
        s = conv_output_size(s, 5, 2, 2)
        s = conv_output_size(s, 3, 1, 2)
        s = conv_output_size(s, 5, 2, 2)
        return s
