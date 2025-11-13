import torch
from torch import nn
from src.efficient_net_v2 import EfficientNetV2
from src.gan.utils import LabelProjection
from src.utils import EfficientNetV2Data

# Parameters
SLOPE = 0.2
EFFNET_OUTPUT_LAYER_SIZE = EfficientNetV2Data.CHANNELS.value[-1]


class EffNetV2Critic(nn.Module):
    # Class that represents a critic network with an efficientNet architecture
    def __init__(self, inp_channels, target_channels=2, image_size=128, proj_dim=50):
        super().__init__()

        # Makes use of the inverted residual block from the efficient net architecture
        self.downsampling_block = EfficientNetV2(inp_channels, image_size=image_size,
                                                 normalization_layer=nn.InstanceNorm2d, activation=nn.LeakyReLU(0.2))
        # Label projection
        self.label_projection = LabelProjection(proj_dim, channels=target_channels, activation=nn.LeakyReLU(0.2))

        # Linear feature transform layer
        self.feature_transform = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=EFFNET_OUTPUT_LAYER_SIZE, out_features=proj_dim),
            nn.LeakyReLU(0.2)
        )

        # Linear output layer
        self.output_layer = nn.Linear(in_features=proj_dim, out_features=1)

    def forward(self, x, y):
        B = x.shape[0]
        # x is image data, y is some conditional data
        x = self.feature_transform(self.downsampling_block(x))
        xp = self.label_projection(x.view(B, -1, 1), y)
        return xp + self.output_layer(x)
