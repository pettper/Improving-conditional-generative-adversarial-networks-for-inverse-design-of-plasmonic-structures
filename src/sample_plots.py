import torch
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import numpy as np
from src.utils import DimerVariable, DimerDataset as Dataset
from torchvision.transforms import Lambda

# Set seed
seed = 23
torch.manual_seed(seed)

# Setup dataloader
root_dir = "data/au_dimer_cylinder_data/featherfiles/"
transform = Lambda(lambda x: (x - x.min()) / (x.max() - x.min()))
dataset = Dataset(root_dir, DimerVariable.SCATTERING_CROSS_SECTION, target_transform=transform)
training_loader = DataLoader(dataset, batch_size=1)

size = 4
row = [0 for i in range(size)] + [1 for i in range(size)] + [2 for i in range(size)] + [3 for i in range(size)]

n_points = 81
wavelength = np.linspace(400, 800, n_points)

for i, sample in enumerate(training_loader):
    iter = i % (size*size)
    if i % (size*size) == 0:
        fig, ax = plt.subplots(size, size)
        im_fig, im_ax = plt.subplots(size, size)
    image, labels = sample

    col = i % size
    ax[row[iter],  col].plot(wavelength, labels[0, :])
    im_ax[row[iter], col].imshow(torch.squeeze(image[:, 0, :, :]))
plt.show()
