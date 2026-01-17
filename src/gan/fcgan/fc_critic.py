import torch
from torch import nn
from src.gan.utils import LabelEmbeddingNetwork


class FullyConnectedCritic(nn.Module):

    def __init__(self, in_channels, target_channels=2, y_dim=81,  image_size=128, proj_dim=50, features=4, lp=True,
                 embed=True, dropout_rate=0.5):
        super().__init__()

        self.image_size = image_size
        self.features = features
        self.lp = lp
        self.embed = embed

        # MLP
        self.mlp = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(in_features=in_channels*image_size*image_size, out_features=64*features),
            nn.ReLU(),
            nn.Linear(in_features=64*features, out_features=features*64),
            nn.ReLU(),
            nn.Linear(in_features=64*features, out_features=64*features),
            nn.ReLU(),
            nn.Linear(in_features=64 * features, out_features=features * 64),
            nn.ReLU(),
            nn.Linear(in_features=64*features, out_features=proj_dim),
        )

        # Label embedding layer
        self.label_embedding = LabelEmbeddingNetwork(proj_dim, channels=target_channels, activation=nn.ReLU(), dropout_rate=dropout_rate)

        # Prepare for addition layer
        self.add_layer = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(in_features=target_channels * y_dim, out_features=target_channels * y_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(in_features=target_channels * y_dim, out_features=proj_dim),
            nn.ReLU()
        )

        # MLP with linear output layer
        self.output_layer = nn.Sequential(
            nn.Linear(in_features=proj_dim, out_features=proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(in_features=proj_dim, out_features=proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(in_features=proj_dim, out_features=1)
        )

    def forward(self, x, y):
        # x -> (B, inp_channels, image_size, image_size)
        # y -> (B, target_channels, y_dim)
        B = x.shape[0]
        if self.lp and self.embed:
            # Down sampling of image x followed by label projection of y and addition
            x = self.mlp(x).view(B, -1, 1)
            e = self.label_embedding(y).view(B, 1, -1)
            # Label projection
            p = torch.bmm(e, x).flatten(start_dim=1)
            x = self.output_layer(x.view(B, -1))
            return x + p
        elif self.lp and (not self.embed):
            # Label projection without embedding network
            y = self.add_layer(y).view(B, 1, -1)  # -> (B, 1, proj_dim)
            x = self.mlp(x).view(B, -1, 1)  # -> (B, proj_dim, 1)
            # Label projection
            p = torch.bmm(y, x).flatten(start_dim=1)
            x = self.output_layer(x.view(B, -1))
            return x + p
        elif (not self.lp) and self.embed:
            # Label embedding network without label projection
            y = self.label_embedding(y)
            x = self.mlp(x)
            x = x + y
            return self.output_layer(x)
        else:
            y = self.add_layer(y)    # -> (B, proj_dim)
            x = self.mlp(x)          # -> (B, proj_dim)
            x = x + y
            return self.output_layer(x)

