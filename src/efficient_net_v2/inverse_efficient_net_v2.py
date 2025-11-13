import torch
from torch import nn
from src.utils.blocks import FusedSETransposedInvertedResidualBlock, SETransposedInvertedResidualBlock
from src.utils import conv_output_size, construct_eff_net_sequence
from src.utils import EfficientNetV2Data

# Parameters
channels = EfficientNetV2Data.CHANNELS.value
layers = EfficientNetV2Data.LAYERS.value
stride = EfficientNetV2Data.STRIDE.value
expansion = EfficientNetV2Data.EXPANSION.value


class InverseEfficientNetV2(nn.Module):

    def __init__(self, out_channels, in_features, normalization_layer=nn.BatchNorm2d, image_size=64,
                 activation=nn.ReLU()):
        super().__init__()
        self.image_size = image_size
        size_in = self.get_avg_pool_size()

        # Reshape input
        self.input_blocks = nn.Sequential(
            nn.Unflatten(dim=1, unflattened_size=(channels[7], 1, 1)),
            nn.Upsample(scale_factor=size_in, mode='bilinear'),
            nn.Conv2d(channels[7], channels[6], kernel_size=1, padding=0)
        )
        # Inverse sequence 1
        self.blocks6 = construct_eff_net_sequence(SETransposedInvertedResidualBlock, channels[5], channels[6],
                                                  stride[6], expansion[5], layers[6], activation, normalization_layer,
                                                  is_transposed=True)
        # Inverse sequence 2
        self.blocks5 = construct_eff_net_sequence(SETransposedInvertedResidualBlock, channels[4], channels[5],
                                                  stride[5], expansion[4], layers[5], activation, normalization_layer,
                                                  is_transposed=True)
        # Inverse sequence 3
        self.blocks4 = construct_eff_net_sequence(SETransposedInvertedResidualBlock, channels[3], channels[4],
                                                  stride[4], expansion[3], layers[4], activation, normalization_layer,
                                                  is_transposed=True)
        # Inverse sequence 4
        self.blocks3 = construct_eff_net_sequence(FusedSETransposedInvertedResidualBlock, channels[2], channels[3],
                                                  stride[3], expansion[2], layers[3], activation, normalization_layer,
                                                  is_transposed=True)
        # Inverse sequence 5
        self.blocks2 = construct_eff_net_sequence(FusedSETransposedInvertedResidualBlock, channels[1], channels[2],
                                                  stride[2], expansion[1], layers[2], activation, normalization_layer,
                                                  is_transposed=True)
        # Inverse sequence 6
        self.blocks1 = construct_eff_net_sequence(FusedSETransposedInvertedResidualBlock, channels[0], channels[1],
                                                  stride[1], expansion[0], layers[1], activation, normalization_layer,
                                                  is_transposed=True)
        # Output layer
        self.output = nn.ConvTranspose2d(channels[0], out_channels, kernel_size=3, stride=stride[0], padding=1,
                                         output_padding=stride[0]-1, bias=False)

    def forward(self, x):
        x = self.input_blocks(x)
        x = self.blocks6(x)
        x = self.blocks5(x)
        x = self.blocks4(x)
        x = self.blocks3(x)
        x = self.blocks2(x)
        x = self.blocks1(x)
        return self.output(x)

    def get_avg_pool_size(self):
        s = conv_output_size(self.image_size, 3, 1, stride[0])
        s = conv_output_size(s, 3, 1, stride[1])
        s = conv_output_size(s, 3, 1, stride[2])
        s = conv_output_size(s, 3, 1, stride[3])
        s = conv_output_size(s, 3, 1, stride[4])
        s = conv_output_size(s, 3, 1, stride[5])
        s = conv_output_size(s, 3, 1, stride[6])
        return s




