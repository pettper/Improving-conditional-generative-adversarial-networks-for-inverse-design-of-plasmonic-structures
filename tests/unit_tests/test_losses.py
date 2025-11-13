import unittest
import torch
from src.utils import MSLELoss, WMSELoss


class TestLosses(unittest.TestCase):

    def test_msle_loss_is_zero_for_equal_tensors(self):
        prediction = torch.Tensor([[1, 2, 3], [4, 5, 6]]).repeat(8, 1, 1, 1)
        target = torch.Tensor([[1, 2, 3], [4, 5, 6]]).repeat(8, 1, 1, 1)
        loss = MSLELoss()

        assert loss(prediction, target) == 0

    def test_msle_loss_non_zero_output(self):
        prediction = torch.Tensor([[1, 2, 3], [4, 5, 6]]).repeat(8, 1, 1, 1)
        target = torch.Tensor([[7, 8, 9], [4, 5, 6]]).repeat(8, 1, 1, 1)
        loss = MSLELoss()

        assert loss(prediction, target) > 0

    def test_msle_loss_correct_value(self):
        prediction = (torch.exp(torch.Tensor([1])) - 1).repeat(8, 1, 64, 64)
        target = torch.Tensor([0]).repeat(8, 1, 64, 64)
        loss = MSLELoss()

        tol = 0.000001
        assert (loss(prediction, target) - 1.0) < tol

    def test_wmse_loss_is_zero_for_equal_tensors(self):
        N = 8
        prediction = torch.Tensor([[1, 2, 3], [4, 5, 6]]).repeat(N, 1, 1, 1)
        target = torch.Tensor([[1, 2, 3], [4, 5, 6]]).repeat(N, 1, 1, 1)
        weights = torch.Tensor([x for x in range(N)])
        loss = WMSELoss()

        assert loss(prediction, target, weights) == 0

    def test_wmse_loss_non_zero_output(self):
        N, C, H, W = 8, 1, 64, 64
        prediction = (torch.exp(torch.Tensor([1])) - 1).repeat(N, C, H, W)
        target = torch.Tensor([0]).repeat(N, C, H, W)
        weights = torch.Tensor([x for x in range(N)])

        loss = WMSELoss()

        assert loss(prediction, target, weights) > 0

    def test_wmse_loss_correct_value(self):
        N, C, H, W = 10, 1, 64, 64
        prediction = torch.Tensor([1]).repeat(N, C, H, W)
        target = torch.Tensor([0]).repeat(N, C, H, W)
        weights = torch.Tensor([x+1 for x in range(N)])

        loss = WMSELoss()

        tol = 0.000001
        assert loss(prediction, target, weights) - 0.1 < tol


if __name__ == '__main__':
    unittest.main()


