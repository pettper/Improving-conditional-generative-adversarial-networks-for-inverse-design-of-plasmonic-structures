import torch
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm as sn
from src.gan.dcgan import DCGANDiscriminator64, DCGANDiscriminator128

# Parameters
SLOPE = 0.2


class DCGANDiscriminator(nn.Module):

    def __init__(self, inp_channels, y_dim, image_size=64, features=4, spectral_normalization=False):
        super().__init__()

        self.image_size = image_size

        if not spectral_normalization:
            def snorm(x):
                return x
        else:
            snorm = sn

        self.down_sampling_block_64 = DCGANDiscriminator64(inp_channels, y_dim, features=features,
                                                           spectral_normalization=spectral_normalization)
        self.down_sampling_block_128 = DCGANDiscriminator128(inp_channels, y_dim, features=features,
                                                             spectral_normalization=spectral_normalization)

        self.label_transform = nn.Sequential(
            snorm(nn.Linear(in_features=y_dim, out_features=y_dim)),
            nn.Unflatten(1, (1, y_dim))
        )

        # Linear output layer
        self.output_layer = nn.Sequential(
            nn.Flatten(start_dim=1),
            snorm(nn.Linear(in_features=y_dim, out_features=1)),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        # x -> (B, y_dim)
        # y -> (B, 2, y_dim)
        if self.image_size == 64:
            x = self.down_sampling_block_64(x)
        elif self.image_size == 128:
            x = self.down_sampling_block_128(x)
        else:
            raise Exception(f"Image size {self.image_size} is not supported")

        y1 = self.label_transform(y[:, 0, :])
        y2 = self.label_transform(y[:, 1, :])

        x1 = torch.bmm(y1, x).flatten()
        x2 = torch.bmm(y2, x).flatten()
        x = self.output_layer(x)

        return self.sigmoid(x1 + x2 + x)
