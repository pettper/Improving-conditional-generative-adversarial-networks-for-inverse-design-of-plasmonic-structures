import torch
from torch import nn


# Mean squared logarithmic error
class MSLELoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, prediction, target):
        return torch.mean((torch.log(1 + prediction) - torch.log(1 + target)) ** 2)
