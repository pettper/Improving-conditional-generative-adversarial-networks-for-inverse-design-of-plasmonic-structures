import torch
from torch import nn
from src.gan.dcgan import DCGANDiscriminator64, DCGANDiscriminator128

# Parameters
SLOPE = 0.2


class DCGANCriticV2(nn.Module):

    def __init__(self, inp_channels, y_dim=81, image_size=128):
        super().__init__()

        self.image_size = image_size
        latent_dim = 20

        self.down_sampling_block_64 = DCGANDiscriminator64(inp_channels, latent_dim)
        self.down_sampling_block_128 = DCGANDiscriminator128(inp_channels, latent_dim)

        self.label_transform = nn.Sequential(
            nn.Linear(in_features=y_dim, out_features=y_dim),
            nn.LeakyReLU(SLOPE),
            nn.Linear(in_features=y_dim, out_features=latent_dim),
            nn.LeakyReLU(SLOPE),
            nn.Unflatten(1, (1, latent_dim))
        )

        # Linear output layer
        self.output_layer = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(in_features=latent_dim, out_features=1),
            nn.LeakyReLU(SLOPE)
        )

    def forward(self, x, y):
        # x -> (B, y_dim)
        # y -> (B, 4, y_dim)
        if self.image_size == 64:
            x = self.down_sampling_block_64(x)
        elif self.image_size == 128:
            x = self.down_sampling_block_128(x)
        else:
            raise Exception(f"Image size {self.image_size} is not supported")

        # Transform the label data for all variables
        y1 = self.label_transform(y[:, 0, :])
        y2 = self.label_transform(y[:, 1, :])
        y3 = self.label_transform(y[:, 2, :])
        y4 = self.label_transform(y[:, 3, :])

        x1 = torch.bmm(y1, x).flatten()
        x2 = torch.bmm(y2, x).flatten()
        x3 = torch.bmm(y3, x).flatten()
        x4 = torch.bmm(y4, x).flatten()
        x = self.output_layer(x)

        return x1 + x2 + x3 + x4 + x
