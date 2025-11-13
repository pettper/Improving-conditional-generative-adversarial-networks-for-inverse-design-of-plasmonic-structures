import unittest
import torch
from src.autoencoder import FullyConnectedAutoencoder, ConvolutionalAutoencoder


class TestAutoencoders(unittest.TestCase):

    def test_fc_autoencoder_forward_method(self):

        B = 16  # batch size
        Y_DIM = 81  # Input dimension

        autoencoder = FullyConnectedAutoencoder(input_dim=Y_DIM)
        x = torch.rand(B, Y_DIM)
        output = autoencoder(x)
        self.assertEqual(x.shape, output.shape)  # add assertion here

    def test_fc_autoencoder_encode_method(self):

        B = 16  # batch size
        Y_DIM = 81  # Input dimension

        autoencoder = FullyConnectedAutoencoder(input_dim=Y_DIM)
        x = torch.rand(B, Y_DIM)
        output = autoencoder.encode(x)
        shape = output.shape

        self.assertEqual(torch.Size([B, 1]), shape)

    def test_conv_autoencoder_forward_method(self):
        B = 16  # batch size
        Y_DIM = 81  # Input dimension

        autoencoder = ConvolutionalAutoencoder(input_dim=Y_DIM)
        x = torch.rand(B, Y_DIM)
        output = autoencoder(x)
        self.assertEqual(x.shape, output.shape)  # add assertion here

    def test_conv_autoencoder_encode_method(self):
        B = 16  # batch size
        Y_DIM = 81  # Input dimension
        CODE_DIM = 4

        autoencoder = ConvolutionalAutoencoder(input_dim=Y_DIM, code_dim=CODE_DIM)
        x = torch.rand(B, Y_DIM)
        output = autoencoder.encode(x)
        shape = output.shape

        self.assertEqual(torch.Size([B, 4]), shape)


if __name__ == '__main__':
    unittest.main()
