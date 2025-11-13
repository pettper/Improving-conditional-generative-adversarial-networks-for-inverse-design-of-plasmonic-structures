import unittest
import torch
from src.cnn_regression import (EfficientNetRegressionV1, EfficientNetRegressionV2, EfficientNetRegressionV3,
                                EfficientNetV2Regression, EfficientNetRegressionV3General,
                                EfficientNetV2RegressionGeneral)


class MyTestCase(unittest.TestCase):

    def test_efficient_net_regression_v1(self):
        y_dim = 20
        inp_channels = 1
        pixels = 64
        B = 8
        model = EfficientNetRegressionV1(inp_channels, y_dim, image_size=pixels)
        x = torch.rand(B, inp_channels, pixels, pixels)
        x = model(x)

        actual_shape = x.shape
        true_shape = torch.Size((B, 2, y_dim))
        self.assertEqual(true_shape, actual_shape)

    def test_efficient_net_regression_v2(self):
        y_dim = 20
        inp_channels = 1
        pixels = 64
        B = 8
        model = EfficientNetRegressionV2(inp_channels, y_dim, image_size=pixels)
        x = torch.rand(B, inp_channels, pixels, pixels)
        x = model(x)

        actual_shape = x.shape
        true_shape = torch.Size((B, 2, y_dim))
        self.assertEqual(true_shape, actual_shape)

    def test_efficient_net_regression_v3(self):
        y_dim = 20
        inp_channels = 2
        pixels = 128
        B = 8
        model = EfficientNetRegressionV3(inp_channels, y_dim, image_size=pixels)
        x = torch.rand(B, inp_channels, pixels, pixels)
        x = model(x)

        actual_shape = x.shape
        true_shape = torch.Size((B, y_dim))
        self.assertEqual(true_shape, actual_shape)

    def test_efficient_net_v2_regression(self):
        y_dim = 20
        inp_channels = 2
        pixels = 128
        B = 8
        model = EfficientNetV2Regression(inp_channels, y_dim, image_size=pixels)
        x = torch.rand(B, inp_channels, pixels, pixels)
        x = model(x)

        actual_shape = x.shape
        true_shape = torch.Size((B, y_dim))
        self.assertEqual(true_shape, actual_shape)

    def test_effnet_general(self):
        y_dim = 20
        inp_channels = 2
        out_channels = 4
        pixels = 128
        B = 8
        model = EfficientNetRegressionV3General(inp_channels, y_dim, image_size=pixels, out_channels=out_channels)
        x = torch.rand(B, inp_channels, pixels, pixels)
        x = model(x)

        actual_shape = x.shape
        true_shape = torch.Size((B, out_channels, y_dim))
        self.assertEqual(true_shape, actual_shape)

    def test_effnetv2_general(self):
        y_dim = 20
        inp_channels = 2
        out_channels = 4
        pixels = 128
        B = 8
        model = EfficientNetV2RegressionGeneral(inp_channels, y_dim, image_size=pixels, out_channels=out_channels)
        x = torch.rand(B, inp_channels, pixels, pixels)
        x = model(x)

        actual_shape = x.shape
        true_shape = torch.Size((B, out_channels, y_dim))
        self.assertEqual(true_shape, actual_shape)


if __name__ == '__main__':
    unittest.main()
