from torch import nn

K = 3
P = 1


class ConvolutionalAutoencoder(nn.Module):
    """
    Class that implements an autoencoder with fully connected hidden layers
    """

    def __init__(self, input_dim, code_dim=1, features=1):
        """
        Creates a convolutional autoencoder object. Accepts only one input channel.
        :param input_dim: The input dimension of x
        :param code_dim: The encoded dimension
        """
        super().__init__()

        # Creates valid dimensions for the 4 hidden layers in the autoencoder
        dim1 = 256 * features
        dim2 = 128 * features
        dim3 = 64 * features
        dim4 = 32 * features

        self.encoder = nn.Sequential(
            nn.Unflatten(1, (1, input_dim)),
            nn.Conv1d(1, dim1, kernel_size=K, stride=2, padding=P),
            nn.ReLU(),
            nn.Conv1d(dim1, dim2, kernel_size=K, stride=2, padding=P),
            nn.ReLU(),
            nn.Conv1d(dim2, dim3, kernel_size=K, stride=2, padding=P),
            nn.ReLU(),
            nn.Conv1d(dim3, dim4, kernel_size=K, stride=2, padding=P),
            nn.ReLU(),
            nn.Conv1d(dim4, code_dim, kernel_size=6, stride=2, padding=0),
            nn.Flatten(start_dim=1),    # Follows the convention that autoencoder code have shape (B, code_dim)
            nn.Sigmoid()
        )
        self.decoder = nn.Sequential(
            nn.Unflatten(dim=1, unflattened_size=(code_dim, 1)),
            nn.ConvTranspose1d(code_dim, dim4, kernel_size=6, stride=2, padding=0),
            nn.ReLU(),
            nn.ConvTranspose1d(dim4, dim3, kernel_size=K, stride=2, padding=P),
            nn.ReLU(),
            nn.ConvTranspose1d(dim3, dim2, kernel_size=K, stride=2, padding=P),
            nn.ReLU(),
            nn.ConvTranspose1d(dim2, dim1, kernel_size=K, stride=2, padding=P),
            nn.ReLU(),
            nn.ConvTranspose1d(dim1, 1, kernel_size=K, stride=2, padding=P),
            nn.Flatten()
        )

    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)
