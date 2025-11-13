import torch
from torch import nn

SLOPE = 0.2


class LabelEmbeddingNetwork(nn.Module):

    def __init__(self, proj_dim, channels=2, activation=nn.LeakyReLU(SLOPE), features=4):
        super().__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.label_transform = nn.Sequential(
            nn.Conv1d(channels, features * 8, 5, stride=2, padding=0),
            nn.BatchNorm1d(features * 8),
            activation,
            nn.Conv1d(features * 8, features * 16, 5, stride=2, padding=2),
            nn.BatchNorm1d(features * 16),
            activation,
            nn.Conv1d(features * 16, features * 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(features * 32),
            activation,
            nn.Conv1d(features * 32, features * 128, kernel_size=3, stride=2, padding=1),
            activation,
        ).to(self.device)

        # Label output layer
        self.label_output = nn.Sequential(
            # OBS! Kernel size = 5, is an adhoc solution for y.shape = (B, C, 81)
            nn.AvgPool1d(kernel_size=5, stride=1, padding=0),
            nn.Flatten(start_dim=1),
            nn.Linear(in_features=features * 128, out_features=proj_dim),
            activation
        ).to(self.device)

    def forward(self, y):
        # Label embedding network forward method. Expects either y -> (B, Y_DIM) or (B, N, Y_DIM) as input.
        # Outputs an embedding vector e -> (B, proj_dim)
        if len(y.shape) == 2:
            e = self.label_transform(y.unsqueeze(1))
            e = self.label_output(e)
        elif len(y.shape) == 3:
            e = self.label_transform(y)
            e = self.label_output(e)
        else:
            raise Exception("Unsupported shape for input y. Expects either 2 or 3 dimensional input.")
        return e

