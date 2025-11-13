import unittest
import torch
from src.gan import Critic, CriticV2
from src.gan.dcgan import DCGANCritic
from src.gan.fcgan import FullyConnectedCritic
from src.gan.effnetgan import EffNetCritic
from src.gan.effnetgan import EffNetV2Critic


# Example command to run from terminal, python -m unittest tests.test_generator.TestGenerator (From root directory).
class TestCritics(unittest.TestCase):

    def test_critic_output_shape(self):
        pixels = 64

        # Create critic
        y_dim = 81
        inp_channels = 1
        critic = Critic(inp_channels, y_dim)

        # Pass input to critic
        x = torch.rand(1, inp_channels, pixels, pixels)
        y = torch.rand(1, y_dim)
        output_tensor = critic(x, y)

        # Test output shape
        shape = output_tensor.shape
        self.assertEqual(1, shape[0])
        self.assertEqual(1, shape[1])

    def test_critic_v2_output_shape(self):
        pixels = 64

        # Create critic
        y_dim = 81
        inp_channels = 1
        critic = CriticV2(inp_channels, y_dim)

        # Pass input to critic
        x = torch.rand(1, inp_channels, pixels, pixels)
        y = torch.rand(1, y_dim)
        output_tensor = critic(x, y)

        # Test output shape
        shape = output_tensor.shape
        self.assertEqual(1, shape[0])
        self.assertEqual(1, shape[1])

    def test_dcgan_critic_output_shape(self):
        pixels = 128

        # Create critic
        y_dim = 81
        inp_channels = 1
        target_channels = 2
        critic = DCGANCritic(inp_channels, target_channels=target_channels, image_size=pixels, lp=True, embed=True)

        # Pass input to critic
        B = 8
        x = torch.rand(B, inp_channels, pixels, pixels)
        y = torch.rand(B, target_channels, y_dim)
        output_tensor = critic(x, y)

        # Test output shape
        shape = output_tensor.shape
        self.assertEqual(B, shape[0])
        self.assertEqual(1, shape[1])

    def test_dcgan_critic_simple(self):
        pixels = 128

        # Create critic
        B = 8
        y_dim = 81
        inp_channels = 2
        target_channels = 2
        critic = DCGANCritic(inp_channels, target_channels=target_channels, image_size=pixels, lp=False, embed=False)

        # Pass input to critic
        x = torch.rand(B, inp_channels, pixels, pixels)
        y = torch.rand(B, target_channels, y_dim)
        output_tensor = critic(x, y)

        # Test output shape
        shape = output_tensor.shape
        self.assertEqual(B, shape[0])
        self.assertEqual(1, shape[1])

    def test_dcgan_critic_only_embedding_network(self):
        pixels = 128

        # Create critic
        B = 8
        y_dim = 81
        inp_channels = 2
        target_channels = 2
        critic = DCGANCritic(inp_channels, target_channels=target_channels, image_size=pixels, lp=False, embed=True)

        # Pass input to critic
        x = torch.rand(B, inp_channels, pixels, pixels)
        y = torch.rand(B, target_channels, y_dim)
        output_tensor = critic(x, y)

        # Test output shape
        shape = output_tensor.shape
        self.assertEqual(B, shape[0])
        self.assertEqual(1, shape[1])

    def test_dcgan_critic_only_label_projection(self):
        pixels = 128

        # Create critic
        B = 8
        y_dim = 81
        inp_channels = 2
        target_channels = 2
        critic = DCGANCritic(inp_channels, target_channels=target_channels, image_size=pixels, lp=True, embed=False)

        # Pass input to critic
        x = torch.rand(B, inp_channels, pixels, pixels)
        y = torch.rand(B, target_channels, y_dim)
        output_tensor = critic(x, y)

        # Test output shape
        shape = output_tensor.shape
        self.assertEqual(B, shape[0])
        self.assertEqual(1, shape[1])

    def test_fcgan_critic_output_shape(self):
        pixels = 128

        # Create critic
        B = 8
        y_dim = 81
        inp_channels = 2
        critic = FullyConnectedCritic(inp_channels, image_size=pixels, lp=True, embed=True)

        # Pass input to critic
        x = torch.rand(B, inp_channels, pixels, pixels)
        y = torch.rand(B, 2, y_dim)
        output_tensor = critic(x, y)

        # Test output shape
        shape = output_tensor.shape
        self.assertEqual(B, shape[0])
        self.assertEqual(1, shape[1])

    def test_fcgan_critic_simple(self):
        pixels = 128

        # Create critic
        B = 8
        y_dim = 81
        inp_channels = 2
        target_channels = 2
        critic = FullyConnectedCritic(inp_channels, target_channels=target_channels, image_size=pixels, lp=False,
                                      embed=False)

        # Pass input to critic
        x = torch.rand(B, inp_channels, pixels, pixels)
        y = torch.rand(B, target_channels, y_dim)
        output_tensor = critic(x, y)

        # Test output shape
        shape = output_tensor.shape
        self.assertEqual(B, shape[0])
        self.assertEqual(1, shape[1])

    def test_fcgan_critic_only_embedding_network(self):
        pixels = 128

        # Create critic
        B = 8
        y_dim = 81
        inp_channels = 2
        target_channels = 2
        critic = FullyConnectedCritic(inp_channels, target_channels=target_channels, image_size=pixels, lp=False,
                                      embed=True)

        # Pass input to critic
        x = torch.rand(B, inp_channels, pixels, pixels)
        y = torch.rand(B, target_channels, y_dim)
        output_tensor = critic(x, y)

        # Test output shape
        shape = output_tensor.shape
        self.assertEqual(B, shape[0])
        self.assertEqual(1, shape[1])

    def test_fcgan_critic_only_label_projection(self):
        pixels = 128

        # Create critic
        B = 8
        y_dim = 81
        inp_channels = 2
        target_channels = 2
        critic = FullyConnectedCritic(inp_channels, target_channels=target_channels, image_size=pixels, lp=True,
                                      embed=False)

        # Pass input to critic
        x = torch.rand(B, inp_channels, pixels, pixels)
        y = torch.rand(B, target_channels, y_dim)
        output_tensor = critic(x, y)

        # Test output shape
        shape = output_tensor.shape
        self.assertEqual(B, shape[0])
        self.assertEqual(1, shape[1])

    def test_effnetgan_critic_output_shape(self):
        pixels = 128

        # Create critic
        B = 8
        y_dim = 81
        inp_channels = 2
        critic = EffNetCritic(inp_channels, image_size=pixels)

        # Pass input to critic
        x = torch.rand(B, inp_channels, pixels, pixels)
        y = torch.rand(B, 2, y_dim)
        output_tensor = critic(x, y)

        # Test output shape
        shape = output_tensor.shape
        self.assertEqual(B, shape[0])
        self.assertEqual(1, shape[1])

    def test_effnetv2gan_critic_output_shape(self):
        pixels = 128

        # Create critic
        B = 8
        y_dim = 81
        inp_channels = 2
        critic = EffNetV2Critic(inp_channels, image_size=pixels)

        # Pass input to critic
        x = torch.rand(B, inp_channels, pixels, pixels)
        y = torch.rand(B, 2, y_dim)
        output_tensor = critic(x, y)

        # Test output shape
        shape = output_tensor.shape
        self.assertEqual(B, shape[0])
        self.assertEqual(1, shape[1])



if __name__ == '__main__':
    unittest.main()
