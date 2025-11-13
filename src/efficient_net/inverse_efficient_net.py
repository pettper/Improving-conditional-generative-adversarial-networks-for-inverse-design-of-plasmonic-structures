from torch import nn
from src.utils.blocks import SETransposedInvertedResidualBlock
from src.utils import conv_output_size

# Parameters
R_EXP = 6  # Expansion factor


class InverseEfficientNet(nn.Module):

    def __init__(self, out_channels, in_features, normalization_layer=nn.BatchNorm2d, image_size=64):
        super().__init__()
        self.image_size = image_size
        size_in = self.get_avg_pool_size()

        # Convolutional part of EfficientNet B0
        self.inverse_efficient_net_b0 = nn.Sequential(
            # Reshape input
            nn.Unflatten(dim=1, unflattened_size=(1280, 1, 1)),
            nn.Upsample(scale_factor=size_in, mode='bilinear'),
            # Reverse sequence 1
            nn.Conv2d(1280, 320, kernel_size=1, padding=0),
            SETransposedInvertedResidualBlock(320, 192, expansion_ratio=R_EXP, kernel_size=3,
                                              stride=1, normalization_layer=normalization_layer),
            # Reverse sequence 2
            SETransposedInvertedResidualBlock(192, 192, expansion_ratio=R_EXP, kernel_size=5,
                                              stride=2, padding=2, normalization_layer=normalization_layer),
            SETransposedInvertedResidualBlock(192, 192, expansion_ratio=R_EXP, kernel_size=5,
                                              stride=1, padding=2, normalization_layer=normalization_layer),
            SETransposedInvertedResidualBlock(192, 192, expansion_ratio=R_EXP, kernel_size=5,
                                              stride=1, padding=2, normalization_layer=normalization_layer),
            SETransposedInvertedResidualBlock(192, 112, expansion_ratio=R_EXP, kernel_size=5,
                                              stride=1, padding=2, normalization_layer=normalization_layer),
            # Reverse sequence 3
            SETransposedInvertedResidualBlock(112, 112, expansion_ratio=R_EXP, kernel_size=5,
                                              stride=1, padding=2, normalization_layer=normalization_layer),
            SETransposedInvertedResidualBlock(112, 112, expansion_ratio=R_EXP, kernel_size=5,
                                              stride=1, padding=2, normalization_layer=normalization_layer),
            SETransposedInvertedResidualBlock(112, 80, expansion_ratio=R_EXP, kernel_size=5,
                                              stride=1, padding=2, normalization_layer=normalization_layer),
            # Reverse sequence 4
            SETransposedInvertedResidualBlock(80, 80, expansion_ratio=R_EXP, kernel_size=3,
                                              stride=2, normalization_layer=normalization_layer),
            SETransposedInvertedResidualBlock(80, 80, expansion_ratio=R_EXP, kernel_size=3,
                                              stride=1, normalization_layer=normalization_layer),
            SETransposedInvertedResidualBlock(80, 40, expansion_ratio=R_EXP, kernel_size=3,
                                              stride=1, normalization_layer=normalization_layer),
            # Reverse sequence 5
            SETransposedInvertedResidualBlock(40, 40, expansion_ratio=R_EXP, kernel_size=5,
                                              stride=2, padding=2, normalization_layer=normalization_layer),
            SETransposedInvertedResidualBlock(40, 24, expansion_ratio=R_EXP, kernel_size=5,
                                              stride=1, padding=2, normalization_layer=normalization_layer),
            # Reverse sequence 6
            SETransposedInvertedResidualBlock(24, 24, expansion_ratio=R_EXP, kernel_size=3,
                                              stride=2, normalization_layer=normalization_layer),
            SETransposedInvertedResidualBlock(24, 16, expansion_ratio=R_EXP, kernel_size=3,
                                              stride=1, normalization_layer=normalization_layer),
            SETransposedInvertedResidualBlock(16, 32, expansion_ratio=1, kernel_size=3,
                                              stride=1, normalization_layer=normalization_layer),
            # Output layer
            nn.ConvTranspose2d(32, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
        )

    def forward(self, x):
        return self.inverse_efficient_net_b0(x)

    def get_avg_pool_size(self):
        s = conv_output_size(self.image_size, 3, 1, 2)
        s = conv_output_size(s, 3, 1, 2)
        s = conv_output_size(s, 5, 2, 2)
        s = conv_output_size(s, 3, 1, 2)
        s = conv_output_size(s, 5, 2, 2)
        return s