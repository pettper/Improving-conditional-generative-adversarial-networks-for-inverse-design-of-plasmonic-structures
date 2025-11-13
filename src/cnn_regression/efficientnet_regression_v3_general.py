from torch import nn
from src.efficient_net import EfficientNet


class EfficientNetRegressionV3General(nn.Module):

    def __init__(self, inp_channels, y_dim, activation=nn.Softplus(), image_size=64, dropout_rate=0.5, out_channels=1):
        super().__init__()

        self.y_dim = y_dim
        self.out_channels = out_channels

        # Convolutional part of EfficientNet B0
        self.efficient_net = EfficientNet(inp_channels, image_size=image_size)

        # Regression part
        self.regression = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(in_features=1280, out_features=3200),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(in_features=3200, out_features=out_channels * y_dim),
            activation
        )

    def forward(self, x):
        x = self.efficient_net(x)
        return self.regression(x).view(-1, self.out_channels, self.y_dim)

    def embedding_map(self, h):
        y = self.regression(h)
        return y

    def get_y_dim(self):
        return self.y_dim
