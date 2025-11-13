import torch
import time
from src.gan.fcgan import FullyConnectedGenerator, FullyConnectedGeneratorSkip
from src.gan.dcgan import DCGANGenerator
from src.gan.effnetgan import EffNetGenerator
from src.gan.effnetgan import EffNetV2Generator

N = 40
BATCH_SIZE = 64
IMAGE_SIZE = 128
Z_DIM = 100
Y_DIM = 80
CHANNELS = 2
z = torch.randn(BATCH_SIZE, 100)
y = torch.randn(BATCH_SIZE, 80)

fc = FullyConnectedGenerator(CHANNELS, target_channels=1, z_dim=Z_DIM, y_dim=Y_DIM, image_size=IMAGE_SIZE)
fc_skip = FullyConnectedGenerator(CHANNELS, target_channels=1, z_dim=Z_DIM, y_dim=Y_DIM, image_size=IMAGE_SIZE)
dc = DCGANGenerator(CHANNELS, target_channels=1, z_dim=Z_DIM, y_dim=Y_DIM, image_size=IMAGE_SIZE)
eff = EffNetGenerator(CHANNELS, z_dim=Z_DIM, y_dim=Y_DIM, image_size=IMAGE_SIZE)
eff_v2 = EffNetV2Generator(CHANNELS, z_dim=Z_DIM, y_dim=Y_DIM, image_size=IMAGE_SIZE)
dc_cbn = DCGANGenerator(CHANNELS, target_channels=1, z_dim=Z_DIM, y_dim=Y_DIM, image_size=IMAGE_SIZE, use_cbn=True)


def test_generator(generator):

    start = time.perf_counter()
    for _ in range(N):
        generator(z, y)
    end = time.perf_counter()
    return (end - start) / N


time_fc = test_generator(fc)
time_fc_skip = test_generator(fc_skip)
time_dc = test_generator(dc)
time_eff = test_generator(eff)
time_eff_v2 = test_generator(eff_v2)
time_dc_cbn = test_generator(dc_cbn)
print("Generator forward pass test results (seconds per forward pass)")
print(f"FC-generator: {time_fc}")
print(f"FC-generator with skip connection: {time_fc_skip}")
print(f"DC-generator: {time_dc}")
print(f"Eff-generator: {time_eff}")
print(f"Eff-generator-v2: {time_eff_v2}")
print(f"DCCbn-generator: {time_dc_cbn}")
