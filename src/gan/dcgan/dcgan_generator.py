from torch import nn
from src.gan.dcgan import DCGANGenerator64, DCGANGenerator128
from src.gan.utils import LabelEmbeddingNetwork


class DCGANGenerator(nn.Module):

    def __init__(self, out_channels, target_channels=2, z_dim=100, y_dim=41, proj_dim=50, features=4, image_size=64,
                 use_cbn=False):
        super().__init__()

        self.z_dim = z_dim
        self.y_dim = y_dim
        self.proj_dim = proj_dim
        self.image_size = image_size
        self.use_cbn = use_cbn

        self.transform_block = nn.Sequential(
            nn.Linear(in_features=proj_dim, out_features=features * 64 * 4 * 4),
            nn.ReLU(),
            nn.Unflatten(1, (features * 64, 4, 4))
        )
        self.noise_layer = nn.Sequential(
            nn.Linear(in_features=z_dim, out_features=proj_dim),
            nn.ReLU(),
        )
        self.label_embedding = LabelEmbeddingNetwork(self.proj_dim, channels=target_channels, activation=nn.ReLU(), dropout_rate=dropout_rate)

        if self.image_size == 64:
            self.upsampling_block = DCGANGenerator64(out_channels)
        elif self.image_size == 128:
            self.upsampling_block = DCGANGenerator128(out_channels, use_cbn=self.use_cbn, embedding_size=self.proj_dim, features=features)
        else:
            raise Exception(f"Image size {self.image_size} is not supported")

    def forward(self, z, y):
        # z is noise, z -> (B, z_dim)
        # y is some conditional data. Either y -> (B, N, y_dim) or (y -> (B, y_dim))

        # Upsampling method depends on if conditional batch normalization is used.
        if self.use_cbn:
            e = self.label_embedding(y)
            z = self.transform_block(self.noise_layer(z))
            return self.upsampling_block(z, e)
        else:
            # Transform input
            z = self.noise_layer(z)
            e = self.label_embedding(y)

            # Addition + upsampling
            e = e + z
            x = self.transform_block(e)
            return self.upsampling_block(x)

    def get_z_dim(self):
        return self.z_dim

    def get_y_dim(self):
        return self.y_dim
