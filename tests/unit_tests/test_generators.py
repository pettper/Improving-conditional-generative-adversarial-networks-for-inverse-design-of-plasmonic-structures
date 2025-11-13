import unittest
import torch
from src.gan import Generator, GeneratorV2
from src.gan.dcgan import DCGANGenerator
from src.gan.fcgan import FullyConnectedGenerator, FullyConnectedGeneratorSkip
from src.gan.effnetgan import EffNetGenerator
from src.gan.effnetgan import EffNetV2Generator


# Example command to run from terminal, python -m unittest tests.test_generator.TestGenerator (From root directory).
class TestGenerators(unittest.TestCase):

    def test_generator_output_shape(self):
        pixels = 64

        # Create generator
        z_dim = 100
        y_dim = 41
        out_channels = 1
        generator = Generator(out_channels=out_channels, z_dim=z_dim, y_dim=y_dim)

        # Pass input to generator
        z = torch.rand(1, z_dim)
        y = torch.rand(1, y_dim)
        output_tensor = generator(z, y)

        # Test output shape
        shape = output_tensor.shape
        self.assertEqual(1, shape[0])
        self.assertEqual(out_channels, shape[1])
        self.assertEqual(pixels, shape[2])
        self.assertEqual(pixels, shape[3])

    def test_generator_v2_output_shape(self):
        pixels = 64

        # Create generator
        z_dim = 100
        y_dim = 41
        out_channels = 1
        generator = GeneratorV2(out_channels=out_channels, z_dim=z_dim, y_dim=y_dim)

        # Pass input to generator
        z = torch.rand(1, z_dim)
        y = torch.rand(1, y_dim)
        output_tensor = generator(z, y)

        # Test output shape
        shape = output_tensor.shape
        self.assertEqual(1, shape[0])
        self.assertEqual(out_channels, shape[1])
        self.assertEqual(pixels, shape[2])
        self.assertEqual(pixels, shape[3])

    def test_dcgan_generator_output_shape(self):
        pixels = 128

        # Create generator
        z_dim = 100
        y_dim = 81
        out_channels = 1
        target_channels = 2
        generator = DCGANGenerator(out_channels=out_channels, target_channels=target_channels, z_dim=z_dim, y_dim=y_dim,
                                   image_size=pixels)

        # Pass input to generator
        z = torch.rand(1, z_dim)
        y = torch.rand(1, target_channels, y_dim)
        output_tensor = generator(z, y)

        # Test output shape
        shape = output_tensor.shape
        print(shape)
        self.assertEqual(1, shape[0])
        self.assertEqual(out_channels, shape[1])
        self.assertEqual(pixels, shape[2])
        self.assertEqual(pixels, shape[3])

    def test_dcgan_generator_with_cbn_output_shape(self):
        pixels = 128

        # Create generator
        z_dim = 100
        y_dim = 81
        out_channels = 1
        target_channels = 2
        generator = DCGANGenerator(out_channels=out_channels, target_channels=target_channels, z_dim=z_dim, y_dim=y_dim,
                                   image_size=pixels, use_cbn=True)

        # Pass input to generator
        z = torch.rand(1, z_dim)
        y = torch.rand(1, target_channels, y_dim)
        output_tensor = generator(z, y)

        # Test output shape
        shape = output_tensor.shape
        self.assertEqual(1, shape[0])
        self.assertEqual(out_channels, shape[1])
        self.assertEqual(pixels, shape[2])
        self.assertEqual(pixels, shape[3])

    def test_fcgan_generator_output_shape_embed_T(self):
        pixels = 128

        # Create generator
        z_dim = 100
        y_dim = 81
        out_channels = 1
        B = 8
        generator = FullyConnectedGenerator(out_channels=out_channels, z_dim=z_dim, y_dim=y_dim, image_size=pixels,
                                            embed=True)

        # Pass input to generator
        z = torch.rand(B, z_dim)
        y = torch.rand(B, 2, y_dim)
        output_tensor = generator(z, y)

        # Test output shape
        shape = output_tensor.shape
        self.assertEqual(B, shape[0])
        self.assertEqual(out_channels, shape[1])
        self.assertEqual(pixels, shape[2])
        self.assertEqual(pixels, shape[3])

    def test_fcgan_generator_output_shape_embed_F(self):
        pixels = 128

        # Create generator
        z_dim = 100
        y_dim = 81
        out_channels = 1
        B = 8
        generator = FullyConnectedGenerator(out_channels=out_channels, z_dim=z_dim, y_dim=y_dim, image_size=pixels,
                                            embed=False)

        # Pass input to generator
        z = torch.rand(B, z_dim)
        y = torch.rand(B, 2, y_dim)
        output_tensor = generator(z, y)

        # Test output shape
        shape = output_tensor.shape
        self.assertEqual(B, shape[0])
        self.assertEqual(out_channels, shape[1])
        self.assertEqual(pixels, shape[2])
        self.assertEqual(pixels, shape[3])

    def test_fcgan_generator_skip_output_shape(self):
        pixels = 128

        # Create generator
        z_dim = 100
        y_dim = 81
        out_channels = 1
        B = 8
        generator = FullyConnectedGeneratorSkip(out_channels=out_channels, z_dim=z_dim, y_dim=y_dim, image_size=pixels)

        # Pass input to generator
        z = torch.rand(B, z_dim)
        y = torch.rand(B, 2, y_dim)
        output_tensor = generator(z, y)

        # Test output shape
        shape = output_tensor.shape
        self.assertEqual(B, shape[0])
        self.assertEqual(out_channels, shape[1])
        self.assertEqual(pixels, shape[2])
        self.assertEqual(pixels, shape[3])

    def test_effnetgan_generator_output_shape(self):
        pixels = 128

        # Create generator
        z_dim = 100
        y_dim = 65
        out_channels = 1
        B = 8
        generator = EffNetGenerator(out_channels=out_channels, z_dim=z_dim, y_dim=y_dim, image_size=pixels)

        # Pass input to generator
        z = torch.rand(B, z_dim)
        y = torch.rand(B, 2, y_dim)
        output_tensor = generator(z, y)

        # Test output shape
        shape = output_tensor.shape
        self.assertEqual(B, shape[0])
        self.assertEqual(out_channels, shape[1])
        self.assertEqual(pixels, shape[2])
        self.assertEqual(pixels, shape[3])

    def test_effnetv2gan_generator_output_shape(self):
        pixels = 128

        # Create generator
        z_dim = 100
        y_dim = 65
        out_channels = 1
        B = 8
        generator = EffNetV2Generator(out_channels=out_channels, z_dim=z_dim, y_dim=y_dim, image_size=pixels)

        # Pass input to generator
        z = torch.rand(B, z_dim)
        y = torch.rand(B, 2, y_dim)
        output_tensor = generator(z, y)

        # Test output shape
        shape = output_tensor.shape
        self.assertEqual(B, shape[0])
        self.assertEqual(out_channels, shape[1])
        self.assertEqual(pixels, shape[2])
        self.assertEqual(pixels, shape[3])


if __name__ == '__main__':
    unittest.main()
