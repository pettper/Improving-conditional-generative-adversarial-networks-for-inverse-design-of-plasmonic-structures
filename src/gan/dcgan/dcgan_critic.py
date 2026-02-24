import torch
from torch import nn

from src.gan.dcgan import DCGANDiscriminator64, DCGANDiscriminator128
from src.gan.utils import LabelEmbeddingNetwork

SLOPE = 0.2


class DCGANCritic(nn.Module):
    def __init__(
        self,
        inp_channels,
        target_channels=2,
        y_dim=81,
        proj_dim=50,
        image_size=128,
        features=4,
        lp=True,
        embed=True,
        dropout_rate=0.5,
    ):
        super().__init__()

        self.image_size = image_size
        self.lp = lp
        self.embed = embed

        if self.image_size == 64:
            self.down_sampling_block = DCGANDiscriminator64(
                inp_channels, proj_dim, features=features
            )
        elif self.image_size == 128:
            self.down_sampling_block = DCGANDiscriminator128(
                inp_channels, proj_dim, features=features
            )
        else:
            raise Exception(f"Image size {self.image_size} is not supported")

        if self.embed:
            self.label_embedding = LabelEmbeddingNetwork(
                proj_dim=proj_dim,
                channels=target_channels,
                activation=nn.LeakyReLU(SLOPE),
                dropout_rate=dropout_rate,
            )

        if not self.embed:
            self.add_layer = nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.Linear(
                    in_features=target_channels * y_dim,
                    out_features=target_channels * y_dim,
                ),
                nn.LeakyReLU(SLOPE),
                nn.Dropout(dropout_rate),
                nn.Linear(in_features=target_channels * y_dim, out_features=proj_dim),
                nn.LeakyReLU(SLOPE),
            )

        # MLP with linear output layer
        self.output_layer = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(in_features=proj_dim, out_features=proj_dim),
            nn.LeakyReLU(SLOPE),
            nn.Dropout(dropout_rate),
            nn.Linear(in_features=proj_dim, out_features=proj_dim),
            nn.LeakyReLU(SLOPE),
            nn.Dropout(dropout_rate),
            nn.Linear(in_features=proj_dim, out_features=1),
        )

    def forward(self, x, y):
        # x -> (B, inp_channels, image_size, image_size)
        # y -> (B, target_channels, y_dim)
        B = x.shape[0]
        # if label projection and label embedding network is used...
        if self.lp and self.embed:
            # Use both label embedding network and label projection
            x = self.down_sampling_block(x)
            e = self.label_embedding(y).view(B, 1, -1)
            # Label projection
            p = torch.bmm(e, x).flatten(start_dim=1)
            x = self.output_layer(x)
            return x + p
        elif self.lp and (not self.embed):
            # Label projection without embedding network
            y = self.add_layer(y).view(B, 1, -1)  # -> (B, 1, proj_dim)
            x = self.down_sampling_block(x)  # -> (B, proj_dim, 1)
            p = torch.bmm(y, x).flatten(start_dim=1)
            x = self.output_layer(x)
            return x + p
        elif (not self.lp) and self.embed:
            # Label embedding network without label projection
            y = self.label_embedding(y)
            x = self.down_sampling_block(x).view(B, -1)
            x = x + y
            return self.output_layer(x)
        else:
            y = self.add_layer(y)  # -> (B, proj_dim)
            x = self.down_sampling_block(x).view(B, -1)  # -> (B, proj_dim, 1)
            x = x + y
            return self.output_layer(x)
