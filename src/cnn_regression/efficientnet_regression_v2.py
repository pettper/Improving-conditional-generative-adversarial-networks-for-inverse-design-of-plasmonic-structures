from torch import nn
import torch
from src.efficient_net import EfficientNet

# Parameters
EFF_NET_OUTPUT_SIZE = 1280
GRU_LAYERS = 1


class EfficientNetRegressionV2(nn.Module):

    def __init__(self, inp_channels, y_dim, image_size=64):
        super().__init__()

        self.y_dim = y_dim

        # Convolutional part of EfficientNet B0
        self.efficient_net = nn.Sequential(
            EfficientNet(inp_channels, image_size=image_size),
            nn.ReLU()
        )

        # 3 gated recurrent units in sequence
        self.h0 = torch.Tensor([1]).repeat(GRU_LAYERS, y_dim)   # Initial hidden state
        self.gru = nn.GRU(EFF_NET_OUTPUT_SIZE, y_dim, num_layers=GRU_LAYERS, batch_first=True)

        # 2 Linear layers
        self.fc_layers = nn.Sequential(
            nn.Linear(in_features=y_dim, out_features=y_dim),
            nn.Linear(in_features=y_dim, out_features=y_dim),
        )

        # output activation
        self.softplus = nn.Softplus()

    def forward(self, x):
        x = self.efficient_net(x)   # -> (B, 1280, 1, 1)

        # GRU blocks
        y1 = self.gru(x.view(-1, EFF_NET_OUTPUT_SIZE), self.h0)[0]  # Unpacks, gru(x) -> outputs, hn
        y2 = self.gru(x.view(-1, EFF_NET_OUTPUT_SIZE), self.h0)[0]

        # Linear layers
        y1 = self.fc_layers(y1)
        y2 = self.fc_layers(y2)

        x = torch.cat((y1.unsqueeze(1), y2.unsqueeze(1)), dim=1)
        return self.softplus(x)

    def get_y_dim(self):
        return self.y_dim
