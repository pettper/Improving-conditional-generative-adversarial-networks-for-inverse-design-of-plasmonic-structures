import torch
from src.gan import Critic, Generator
from src.utils import count_parameters

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
critic = Critic(1, 41).to(device)
generator = Generator(1).to(device)

print(count_parameters(generator))
print(count_parameters(critic))

x = torch.rand(1, 1, 64, 64)
y = torch.rand(1, 41)
z = torch.rand(1, 100)

gen_output = generator(z, y)
critic_output = critic(x, y)

print(gen_output.shape)
print(critic_output.shape)
