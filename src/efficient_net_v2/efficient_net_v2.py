from torch import nn
from src.utils.blocks import FusedSEInvertedResidualBlock, SEInvertedResidualBlock
from src.utils import conv_output_size, construct_eff_net_sequence
from src.utils import EfficientNetV2Data

# Parameters
channels = EfficientNetV2Data.CHANNELS.value
layers = EfficientNetV2Data.LAYERS.value
stride = EfficientNetV2Data.STRIDE.value
expansion = EfficientNetV2Data.EXPANSION.value


class EfficientNetV2(nn.Module):

    def __init__(self, inp_channels, normalization_layer=nn.BatchNorm2d, image_size=128, activation=nn.ReLU()):
        super().__init__()
        self.image_size = image_size
        avg_pool_size = self.get_avg_pool_size()

        # Input layer
        self.conv1 = nn.Conv2d(inp_channels, channels[0], kernel_size=3, stride=stride[0], padding=1, bias=False)
        # Sequence 1 blocks
        self.blocks1 = construct_eff_net_sequence(FusedSEInvertedResidualBlock, channels[0], channels[1], stride[1],
                                                  expansion[0], layers[1], activation, normalization_layer)
        # Sequence 2 blocks
        self.blocks2 = construct_eff_net_sequence(FusedSEInvertedResidualBlock, channels[1], channels[2], stride[2],
                                                  expansion[1], layers[2], activation, normalization_layer)
        # Sequence 3 blocks
        self.blocks3 = construct_eff_net_sequence(FusedSEInvertedResidualBlock, channels[2], channels[3], stride[3],
                                                  expansion[2], layers[3], activation, normalization_layer)
        # Sequence 4 blocks
        self.blocks4 = construct_eff_net_sequence(SEInvertedResidualBlock, channels[3], channels[4], stride[4],
                                                  expansion[3], layers[4], activation, normalization_layer)
        # Sequence 5 blocks
        self.blocks5 = construct_eff_net_sequence(SEInvertedResidualBlock, channels[4], channels[5], stride[5],
                                                  expansion[4], layers[5], activation, normalization_layer)
        # Sequence 6 blocks
        self.blocks6 = construct_eff_net_sequence(SEInvertedResidualBlock, channels[5], channels[6], stride[6],
                                                  expansion[5], layers[6], activation, normalization_layer)
        # Output block
        self.output = nn.Sequential(
            nn.Conv2d(channels[6], channels[7], kernel_size=1, stride=1, padding=0),
            nn.AvgPool2d(kernel_size=avg_pool_size, stride=1, padding=0),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.blocks1(x)
        x = self.blocks2(x)
        x = self.blocks3(x)
        x = self.blocks4(x)
        x = self.blocks5(x)
        x = self.blocks6(x)
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




