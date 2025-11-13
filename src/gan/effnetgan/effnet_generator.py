from torch import nn
from src.efficient_net import InverseEfficientNet


class EffNetGenerator(nn.Module):

    def __init__(self, out_channels, z_dim=100, y_dim=81, image_size=128):
        super().__init__()

        self.z_dim = z_dim
        self.y_dim = y_dim
        self.image_size = image_size

        self.upsampling_block = InverseEfficientNet(out_channels, z_dim, image_size=image_size)
        self.label_input_layer = nn.Sequential(
            nn.Linear(in_features=y_dim, out_features=z_dim),
            nn.ReLU()
        )
        self.upsampling_input_layer = nn.Sequential(
            nn.Linear(in_features=z_dim, out_features=1280),
        )

    def forward(self, z, y):
        # z is noise, z -> (B, z_dim)
        # y is some conditional data. Either y -> (B, N, y_dim) or (y -> (B, y_dim))

        # Addition + Naive label input
        if len(y.shape) == 2:
            z = z + self.label_input_layer(y)
        elif len(y.shape) == 3:
            for i in range(y.shape[1]):
                z = z + self.label_input_layer(y[:, i, :])
        else:
            raise Exception("Unsupported shape for input y. Expects either 2 or 3 dimensional input.")

        return self.upsampling_block(self.upsampling_input_layer(z))

    def get_z_dim(self):
        return self.z_dim

    def get_y_dim(self):
        return self.y_dim
