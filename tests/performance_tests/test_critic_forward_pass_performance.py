import torch
import time
from src.gan.fcgan import FullyConnectedCritic
from src.gan.dcgan import DCGANCritic
from src.gan.effnetgan import EffNetCritic
from src.gan.effnetgan import EffNetV2Critic

N = 40
BATCH_SIZE = 64
IMAGE_SIZE = 128
CHANNELS = 2
x = torch.randn(BATCH_SIZE, CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
y = torch.randn(BATCH_SIZE, 80)

fc = FullyConnectedCritic(CHANNELS, target_channels=1, image_size=IMAGE_SIZE)
dc = DCGANCritic(CHANNELS, target_channels=1, image_size=IMAGE_SIZE)
eff = EffNetCritic(CHANNELS, target_channels=1, image_size=IMAGE_SIZE)
eff_v2 = EffNetV2Critic(CHANNELS, target_channels=1, image_size=IMAGE_SIZE)

def test_critic(critic):

    start = time.perf_counter()
    for _ in range(N):
        d = critic(x, y)
    end = time.perf_counter()
    return (end - start) / N


time_fc = test_critic(fc)
time_dc = test_critic(dc)
time_eff = test_critic(eff)
time_eff_v2 = test_critic(eff_v2)
print("Critic forward pass test results (seconds per forward pass)")
print(f"FC-critic: {time_fc}")
print(f"DC-critic: {time_dc}")
print(f"Eff-critic: {time_eff}")
print(f"EffV2-critic: {time_eff_v2}")
