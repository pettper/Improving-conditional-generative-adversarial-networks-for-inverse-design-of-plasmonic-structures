from torch import nn
from src.efficient_net import EfficientNet


class EfficientNetRegressionV3(nn.Module):

    def __init__(self, inp_channels, y_dim, activation=nn.Softplus(), image_size=64, dropout_rate=0.5):
        super().__init__()

        self.y_dim = y_dim

        # Convolutional part of EfficientNet B0
        self.efficient_net = EfficientNet(inp_channels, image_size=image_size)

        # Regression part
        self.regression = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(in_features=1280, out_features=3200),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(in_features=3200, out_features=y_dim),
            activation
        )

    def forward(self, x):
        x = self.efficient_net(x)
        return self.regression(x)

    """
    def forward(self, x, transform=None):
        x = self.efficient_net(x)
        if transform is not None:
            return transform(self.regression(x))
        else:
            return self.regression(x)
    """

    def embedding_map(self, h):
        y = self.regression(h)
        return y

    def get_y_dim(self):
        return self.y_dim
