from src.cnn_regression import EfficientNetV2RegressionGeneral as RegressionModel
from src.utils import DimerVariable, DimerDataset as Dataset
from src.gan.fcgan.fc_generator import FullyConnectedGenerator as Generator
from src.gan.dcgan.dcgan_generator import DCGANGenerator

from pathlib import Path
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

# Save figures directory
fig_dir = "figures/"
Path(fig_dir).mkdir(parents=True, exist_ok=True)

# Parameters
INPUT_CHANNELS = 2
OUTPUT_CHANNELS = 2
Y_DIM = 81
Z_DIM = 100
BATCH_SIZE = 16
IMAGE_SIZE = 128
DROP_OUT = 0.5
variable = DimerVariable.CROSS_SECTIONS

# Dataset
root_dir = Path("../dimer_cylinder_train_val_test")
test_dir = root_dir.joinpath(Path("test/featherfiles/"))
test_dataset = Dataset(test_dir, variable)
test_loader = DataLoader(test_dataset, batch_size=1000, pin_memory=True)


images, labels = next(iter(test_loader))
lambda_min, lambda_max = 400, 800
wavelength = np.linspace(lambda_min, lambda_max, Y_DIM)
idx = 56

# Plot cross section spectra
fig = plt.figure()
plt.plot(wavelength, np.array(labels[idx][0].detach()), linestyle="-", color="red", label="Scattering cross section")
plt.plot(wavelength, np.array(labels[idx][1].detach()), linestyle="--", color="black", label="Absorption cross section")
plt.legend(fontsize=14, loc="upper right")
plt.xlabel("Wavelength [nm]", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.ylim([0.0, 0.6])
fig.savefig(fig_dir + "overview_cross_section.png", format='png', dpi=100, bbox_inches='tight')

# Plot images data
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

cmap = plt.get_cmap('inferno')
cmap = truncate_colormap(cmap, 0.90, 0.0, 500)

fig = plt.figure()
plt.imshow(images[idx, 1, :, :], vmin=-1, vmax=1, cmap=cmap)
plt.colorbar()
im_axes = plt.gca()
plt.setp(im_axes, xticks=[], yticks=[])
plt.title("Dimer", fontsize=14)
fig.savefig(fig_dir + "overview_image.png", format='png', dpi=100, bbox_inches='tight')
