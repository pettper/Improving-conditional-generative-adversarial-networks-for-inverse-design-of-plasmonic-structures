import torch
from torch import nn


# Weighted mean squared error
class WMSELoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, prediction, target, weights):
        size = target[0].flatten().shape[0]
        # Weights must be of the same length as the number of samples in prediction/target

        e = (prediction - target) ** 2
        w = weights.repeat_interleave(size).reshape_as(target)
        s = weights.sum().expand_as(w)
        weighted_mean_squared_error = w * e / s

        return weighted_mean_squared_error.mean()

