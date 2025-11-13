from torch import nn
import numpy as np


class FullyConnectedAutoencoder(nn.Module):
    """
    Class that implements an autoencoder with fully connected hidden layers
    """

    def __init__(self, input_dim, code_dim=1, features=1):
        """
        Creates a fc-autoencoder object.
        :param input_dim: The input dimension of x
        :param code_dim: The encoded dimension
        """
        super().__init__()

        # Creates valid dimensions for the hidden layers in the autoencoder
        dim1 = 256 * features
        dim2 = 128 * features
        dim3 = 64 * features
        dim4 = 32 * features
        # dim5 = 16 * features
        # dim6 = 8 * features
        # dim7 = 4 * features

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, dim1),
            nn.ReLU(),
            nn.Linear(dim1, dim2),
            nn.ReLU(),
            nn.Linear(dim2, dim3),
            nn.ReLU(),
            nn.Linear(dim3, dim4),
            nn.ReLU(),
            nn.Linear(dim4, code_dim),
            nn.Sigmoid()
        )
        """
        nn.ReLU(),
        nn.Linear(dim5, dim6),
        nn.ReLU(),
        nn.Linear(dim6, dim7),
        nn.ReLU(),
        nn.Linear(dim7, code_dim),
        """

        """
        nn.Linear(code_dim, dim7),
        nn.ReLU(),
        nn.Linear(dim7, dim6),
        nn.ReLU(),
        nn.Linear(dim6, dim5),
        nn.ReLU(),
        """

        self.decoder = nn.Sequential(
            nn.Linear(code_dim, dim4),
            nn.ReLU(),
            nn.Linear(dim4, dim3),
            nn.ReLU(),
            nn.Linear(dim3, dim2),
            nn.ReLU(),
            nn.Linear(dim2, dim1),
            nn.ReLU(),
            nn.Linear(dim1, input_dim)
        )

    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)
