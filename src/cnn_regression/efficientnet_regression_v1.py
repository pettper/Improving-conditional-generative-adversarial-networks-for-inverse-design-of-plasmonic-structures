from torch import nn
from src.efficient_net import EfficientNet


class EfficientNetRegressionV1(nn.Module):

    def __init__(self, inp_channels, y_dim, image_size=64, dropout_rate=0.2):
        super().__init__()

        self.y_dim = y_dim

        # Convolutional part of EfficientNet B0
        self.efficient_net = EfficientNet(inp_channels, image_size=image_size)

        # Regression part
        self.regression_v2 = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(in_features=(1280 * 1 * 1), out_features=2 * y_dim),
            nn.Unflatten(1, (2, y_dim)),
            nn.Softplus()
        )

    def forward(self, x):
        x = self.efficient_net(x)
        return self.regression_v2(x)

    def embedding_map(self, h):
        y = self.regression_v2(h)
        return y

    def get_y_dim(self):
        return self.y_dim
