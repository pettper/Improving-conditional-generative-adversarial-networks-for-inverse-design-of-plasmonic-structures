import unittest
import torch
from src.efficient_net import EfficientNet, InverseEfficientNet
from src.efficient_net_v2 import EfficientNetV2, InverseEfficientNetV2
from src.utils import count_parameters


class TestEfficientNet(unittest.TestCase):

    def test_efficient_net_output_size(self):
        pixels1 = 64
        pixels2 = 128
        inp_channels = 1
        tensor1 = torch.rand(1, inp_channels, pixels1, pixels1)
        tensor2 = torch.rand(1, inp_channels, pixels2, pixels2)

        model = EfficientNet(inp_channels, image_size=pixels2)
        count_parameters(model)
        y = model(tensor2)
        self.assertTrue(y.shape == torch.Size((1, 1280, 1, 1)))

        model = EfficientNet(inp_channels, image_size=pixels1)
        y = model(tensor1)
        self.assertTrue(y.shape == torch.Size((1, 1280, 1, 1)))

    def test_inverse_efficient_net_output_size(self):
        out_channels = 1
        batch = 16
        pixels1 = 64
        pixels2 = 128
        in_features = 1280
        tensor1 = torch.rand(batch, in_features)
        tensor2 = torch.rand(batch, in_features)

        model = InverseEfficientNet(out_channels, in_features, image_size=pixels2)
        count_parameters(model)
        y = model(tensor2)
        self.assertEqual(y.shape, torch.Size((batch, out_channels, pixels2, pixels2)))

        model = InverseEfficientNet(out_channels, in_features, image_size=pixels1)
        y = model(tensor1)
        self.assertEqual(y.shape, torch.Size((batch, out_channels, pixels1, pixels1)))

    def test_efficient_net_v2_output_size(self):
        pixels1 = 64
        pixels2 = 128
        inp_channels = 1
        tensor1 = torch.rand(1, inp_channels, pixels1, pixels1)
        tensor2 = torch.rand(1, inp_channels, pixels2, pixels2)

        model = EfficientNetV2(inp_channels, image_size=pixels2)
        print(count_parameters(model))
        y = model(tensor2)
        self.assertTrue(y.shape == torch.Size((1, 1280, 1, 1)))

        model = EfficientNetV2(inp_channels, image_size=pixels1)
        y = model(tensor1)
        self.assertTrue(y.shape == torch.Size((1, 1280, 1, 1)))

    def test_inverse_efficient_net_v2_output_size(self):
        out_channels = 1
        batch = 16
        pixels1 = 64
        pixels2 = 128
        in_features = 1280
        tensor1 = torch.rand(batch, in_features)
        tensor2 = torch.rand(batch, in_features)

        model = InverseEfficientNetV2(out_channels, in_features, image_size=pixels2)
        print(count_parameters(model))
        y = model(tensor2)
        self.assertEqual(y.shape, torch.Size((batch, out_channels, pixels2, pixels2)))

        model = InverseEfficientNetV2(out_channels, in_features, image_size=pixels1)
        y = model(tensor1)
        self.assertEqual(y.shape, torch.Size((batch, out_channels, pixels1, pixels1)))


if __name__ == '__main__':
    unittest.main()
