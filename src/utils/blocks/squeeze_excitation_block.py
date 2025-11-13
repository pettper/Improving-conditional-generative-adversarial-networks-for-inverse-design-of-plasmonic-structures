from torch import nn


class SqueezeExcitationBlock(nn.Module):
    # Code from a squeeze excitation block from the paper https://arxiv.org/pdf/1709.01507.pdf.
    def __init__(self, inp_channels, reduced_dim):
        super().__init__()

        self.se_block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # C x H x W -> C x 1 x 1
            nn.Conv2d(inp_channels, reduced_dim, 1),
            nn.ReLU(),
            nn.Conv2d(reduced_dim, inp_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.se_block(x)
