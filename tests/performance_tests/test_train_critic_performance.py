import torch
from torch.optim import Adam
import time
from src.gan.fcgan import FullyConnectedCritic, FullyConnectedGenerator
from src.gan.dcgan import DCGANCritic, DCGANGenerator
from src.gan.effnetgan import EffNetCritic, EffNetGenerator, EffNetV2Critic, EffNetV2Generator
from src.gan import WGANTrainer

N = 10
BATCH_SIZE = 32
IMAGE_SIZE = 128
CHANNELS = 2
Z_DIM = 100
Y_DIM = 80
LEARNING_RATE = 1e-4
B1 = 0.0
B2 = 0.9
x = torch.randn(BATCH_SIZE, CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
y = torch.randn(BATCH_SIZE, 80)

# Critics
fc_critic = FullyConnectedCritic(CHANNELS, target_channels=1, image_size=IMAGE_SIZE)
dc_critic = DCGANCritic(CHANNELS, target_channels=1, image_size=IMAGE_SIZE)
eff_critic = EffNetCritic(CHANNELS, target_channels=1, image_size=IMAGE_SIZE)
eff_v2_critic = EffNetV2Critic(CHANNELS, target_channels=1, image_size=IMAGE_SIZE)

# Generators
fc_gen = FullyConnectedGenerator(CHANNELS, target_channels=1, z_dim=Z_DIM, y_dim=Y_DIM, image_size=IMAGE_SIZE)
dc_gen = DCGANGenerator(CHANNELS, target_channels=1, z_dim=Z_DIM, y_dim=Y_DIM, image_size=IMAGE_SIZE)
eff_gen = EffNetGenerator(CHANNELS, z_dim=Z_DIM, y_dim=Y_DIM, image_size=IMAGE_SIZE)
eff_v2_gen = EffNetV2Generator(CHANNELS, z_dim=Z_DIM, y_dim=Y_DIM, image_size=IMAGE_SIZE)
dc_cbn_gen = DCGANGenerator(CHANNELS, target_channels=1, z_dim=Z_DIM, y_dim=Y_DIM, image_size=IMAGE_SIZE, use_cbn=True)


def test_train_critic(critic, generator):
    critic_optim = Adam(critic.parameters(), lr=LEARNING_RATE, betas=(B1, B2))
    generator_optim = Adam(generator.parameters(), lr=LEARNING_RATE, betas=(B1, B2))
    trainer = WGANTrainer(critic, generator, critic_optim, generator_optim, None, None,
                          torch.device('cpu'))
    start = time.perf_counter()
    for i in range(N):
        trainer.train_critic(x, y)
    end = time.perf_counter()
    return (end - start) / N


time_fc = test_train_critic(fc_critic, fc_gen)
time_dc = test_train_critic(dc_critic, dc_gen)
time_eff = test_train_critic(eff_critic, eff_gen)
time_eff_v2 = test_train_critic(eff_v2_critic, eff_v2_gen)
time_dc_cbn = test_train_critic(dc_critic, dc_cbn_gen)
print("WGAN-GP train critic test results (seconds per forward pass)")
print(f"FC: {time_fc}")
print(f"DC: {time_dc}")
print(f"Eff: {time_eff}")
print(f"EffV2: {time_eff_v2}")
print(f"DCCbn: {time_dc_cbn}")

