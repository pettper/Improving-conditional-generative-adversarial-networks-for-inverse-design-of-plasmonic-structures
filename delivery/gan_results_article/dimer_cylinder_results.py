# My code
from src.cnn_regression import EfficientNetV2RegressionGeneral as RegressionModel
from src.utils import DimerVariable, DimerDataset as Dataset
from plotting_routines.plots import gan_plotter, gan_prediction_plot2, gan_prediction_plot, gan_prediction_plot3
from src.gan.fcgan.fc_generator import FullyConnectedGenerator as Generator
from src.gan.dcgan.dcgan_generator import DCGANGenerator

# Packages
import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.nn import Softplus
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path


def central_moving_average(arr_epoch, arr, window_size):

    half_window = window_size // 2
    rem_window = window_size % 2
    cma = []
    i = half_window
    while (i+half_window+rem_window) < len(arr):
        cma.append([arr_epoch[i], np.mean(arr[(i - half_window):(i + half_window + rem_window)])])
        i += 1
    return np.array(cma)


# Save figures directory
fig_dir = "figures/dimer_cylinder_structures/"
Path(fig_dir).mkdir(parents=True, exist_ok=True)

# Set seed
torch.manual_seed(23)

# Parameters
INPUT_CHANNELS = 2
OUTPUT_CHANNELS = 2
Y_DIM = 81
Z_DIM = 100
BATCH_SIZE = 16
IMAGE_SIZE = 128
DROP_OUT = 0.5
variable = DimerVariable.CROSS_SECTIONS

# Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
forward_net = RegressionModel(INPUT_CHANNELS, Y_DIM, activation=Softplus(), image_size=IMAGE_SIZE,
                              dropout_rate=DROP_OUT, out_channels=OUTPUT_CHANNELS).to(device)

# Setup dataloader
root_dir = Path("../dimer_cylinder_train_val_test")
train_dir = root_dir.joinpath(Path("training/featherfiles/"))
val_dir = root_dir.joinpath(Path("validation/featherfiles/"))
test_dir = root_dir.joinpath(Path("test/featherfiles/"))

training_dataset = Dataset(train_dir, variable)
validation_dataset = Dataset(val_dir,  # Validation set uses same transforms as in the training set
                             variable,
                             transform=lambda x: training_dataset.apply_transform(x),
                             target_transform=lambda x: training_dataset.apply_target_transform(x))
test_dataset = Dataset(test_dir,  # Test set uses same transforms as in the training set
                       variable,
                       transform=lambda x: training_dataset.apply_transform(x),
                       target_transform=lambda x: training_dataset.apply_target_transform(x))

training_loader = DataLoader(training_dataset, batch_size=BATCH_SIZE, sampler=RandomSampler(training_dataset),
                             pin_memory=True)
validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, sampler=RandomSampler(validation_dataset),
                               pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=1000, pin_memory=True)  # sampler=RandomSampler(test_dataset)

################ PLOTS FOR CNN STARTS HERE ##########################
# Load trained model
checkpoint = torch.load("cnn_results/last_epoch_effv2_cross_lr00001_drop05.pth.tar", map_location=device)
forward_net.load_state_dict(checkpoint['model_state_dict'])
forward_net.eval()

loss = checkpoint['losses']
metric = checkpoint['metric_values']

epochs = loss[:, 0]
train_loss = loss[:, 1]
val_loss = loss[:, 2]

# Plot loss
step = 75
plt.figure()
plt.semilogy(epochs[0::step], train_loss[0::step], label='Training set')
plt.semilogy(epochs[0::step], val_loss[0::step], label='Validation set')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.grid()

images, labels = next(iter(validation_loader))
with torch.no_grad():
    output = forward_net(images)
ylabels = ['Scat. cross sec.', 'Abs cross sec.']

for var in range(2):
    fig, ax = plt.subplots(int(np.sqrt(BATCH_SIZE)), int(np.sqrt(BATCH_SIZE)), figsize=(10, 10))

    lambda_min, lambda_max = 400, 800
    wavelength = np.linspace(lambda_min, lambda_max, Y_DIM)
    it = 0
    i = 0
    while it < BATCH_SIZE:
        j = it % int(np.sqrt(BATCH_SIZE))
        ax[i][j].plot(wavelength, np.array(labels[it, var].detach()), label='FEM')
        ax[i][j].plot(wavelength, np.array(output[it, var].detach()), label='Pred.')
        ax[i][j].set_xlabel('Wavelength [nm]')
        ax[i][j].set_ylabel(ylabels[var])
        ax[i][j].legend()

        if j == (int(np.sqrt(BATCH_SIZE)) - 1):
            i += 1
        it += 1

################# PLOTS FOR GAN STARTS HERE #####################

def extract_gan_plot_data(filename):
    checkpoint = torch.load(filename, map_location=device)
    return checkpoint['reconstruction_error'], checkpoint['forward_error'], checkpoint['wasserstein_distance']


filenames = ["fcgan_results/wgan_fc_cross_cylinder_FF.pth.tar",
             "fcgan_results/wgan_fc_cross_cylinder_TF.pth.tar",
             "fcgan_results/wgan_fc_cross_cylinder_FT.pth.tar",
             "fcgan_results/wgan_fc_cross_cylinder_TT.pth.tar",
             "dcgan_results/wgan_dc_cross_cylinder_FF.pth.tar",
             "dcgan_results/wgan_dc_cross_cylinder_TF.pth.tar",
             "dcgan_results/wgan_dc_cross_cylinder_FT.pth.tar",
             "dcgan_results/wgan_dc_cross_cylinder_TT.pth.tar"
             ]
legends = ["FCGAN",
           "FCGAN + LP",
           "FCGAN + Embed.",
           "FCGAN + LP + Embed.",
           "DCGAN",
           "DCGAN + LP",
           "DCGAN + Embed.",
           "DCGAN + LP + Embed"]
colors = ["blue",
          "green",
          "orange",
          "red",
          "blue",
          "green",
          "orange",
          "red"]
markers = ['o','s','^','d','o','s','^','d']

# Plot parameters
lw = 2.0
ls = ['solid', 'dotted', 'dashed', (0, (3, 1, 1, 1, 1, 1))]
dc_offset = 4


def init_data_plot(ylabels):
    fig, ax = plt.subplots(2, 2, figsize=(12, 12), sharey=True, sharex=True)
    for i in range(2):
        for j in range(2):
            if i == 1:
                ax[i, j].set_xlabel("Epoch", fontsize=16)
            if j == 0:
                ax[i, j].set_ylabel(ylabels[i], fontsize=16)
            ax[i, j].grid()
    return fig, ax


def append_to_data_plot(ax, xdata, ydata, plot_label, plot_color, marker, step=5):
    cma = central_moving_average(xdata, ydata, 15)
    ax.semilogy(cma[0::step, 0], cma[0::step, 1], label=plot_label, linewidth=lw, marker=marker, markersize=4,
                linestyle='solid', color=plot_color, markeredgecolor=plot_color, markerfacecolor='None')
    ax.legend(fontsize=14)
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)


# Plot error estimates, Reconstruction error and CNN-error
fig, ax = init_data_plot([r"$\mathrm{MAE}_{image}$", "$\mathrm{MAE}_{spectra}$"])
for i, f in enumerate(filenames):
    rce, _, _ = extract_gan_plot_data(f)  # (N, 5) with entries (Epoch, train rce mean, train rce var, val rce mean, val rce var)
    if i < dc_offset:
        # Special case due different scales on x-axis after retraining fcgan models
        rce = np.append(rce[0:1500][0::5], rce[1500:], axis=0)
        append_to_data_plot(ax[0, 0], rce[:, 0], rce[:, 3], legends[i], colors[i], markers[i])
    else:
        append_to_data_plot(ax[0, 1], rce[:, 0], rce[:, 3], legends[i], colors[i], markers[i])

for i, f in enumerate(filenames):
    #  (N, 5) with entries (Epoch, train forward error, train cnn error, val forward error, val cnn error)
    _, fwde, _ = extract_gan_plot_data(f)
    if i < dc_offset:
        # Special case due different scales on x-axis after retraining fcgan models
        fwde = np.append(fwde[0:1500][0::5], fwde[1500:], axis=0)
        append_to_data_plot(ax[1, 0], fwde[:, 0], fwde[:, 4], legends[i], colors[i], markers[i])
    else:
        append_to_data_plot(ax[1, 1], fwde[:, 0], fwde[:, 4], legends[i], colors[i], markers[i])
ax[0, 0].annotate('a)', xy=(0.15, 0.9), xycoords="axes fraction", fontsize=18)
ax[0, 1].annotate('b)', xy=(0.15, 0.9), xycoords="axes fraction", fontsize=18)
ax[1, 0].annotate('c)', xy=(0.15, 0.9), xycoords="axes fraction", fontsize=18)
ax[1, 1].annotate('d)', xy=(0.15, 0.9), xycoords="axes fraction", fontsize=18)
fig.subplots_adjust(wspace=0.03, hspace=0.03)
fig.savefig(fig_dir + "gan_error_plots.png", format='png', dpi=150, bbox_inches='tight')
fig.savefig(fig_dir + "gan_error_plots.svg", format='svg', dpi=100, bbox_inches='tight')

plt.figure()
for i, f in enumerate(filenames):
    _, _, wd = extract_gan_plot_data(f)
    wasserstein_distance = torch.Tensor(wd).detach()
    plt.semilogy(wasserstein_distance[0::step, 0], wasserstein_distance[0::step, 1], label=legends[i])
    plt.xlabel('Epoch')
    plt.ylabel('Wasserstein distance')
    plt.legend()
    plt.grid()

##################### GAN Test set plots ########################
# FCGAN
generator = Generator(INPUT_CHANNELS, target_channels=variable.size(), y_dim=Y_DIM, z_dim=Z_DIM,
                      image_size=IMAGE_SIZE).to(device)
generator.eval()
generator2 = Generator(INPUT_CHANNELS, target_channels=variable.size(), y_dim=Y_DIM, z_dim=Z_DIM,
                      image_size=IMAGE_SIZE).to(device)
generator2.eval()
#DCGAN
dc_generator = DCGANGenerator(INPUT_CHANNELS, target_channels=variable.size(), y_dim=Y_DIM, z_dim=Z_DIM,
                              image_size=IMAGE_SIZE, use_cbn=False).to(device)
dc_generator.eval()
dc_generator2 = DCGANGenerator(INPUT_CHANNELS, target_channels=variable.size(), y_dim=Y_DIM, z_dim=Z_DIM,
                              image_size=IMAGE_SIZE, use_cbn=False).to(device)
dc_generator2.eval()

fig_names = ["fcgan.png",
             "fcgan_lp.png",
             "fcgan_embed.png",
             "fcgan_lp_embed.png",
             "dcgan.png",
             "dcgan_lp.png",
             "dcgan_embed.png",
             "dcgan_lp_embed.png"]

for i, f in enumerate(filenames):
    checkpoint = torch.load(f, map_location=device)

    if i < 4:
        generator.load_state_dict(checkpoint['generator_state_dict'])
        fig = gan_plotter(generator, test_loader, legends[i])
    else:
        dc_generator.load_state_dict(checkpoint['generator_state_dict'])
        fig = gan_plotter(dc_generator, test_loader, legends[i])
    fig.savefig(fig_dir + fig_names[i], format='png', dpi=100, bbox_inches='tight')
    #gan_prediction_plotter(generator, forward_net, test_loader)


########## Custom plot from GAN ############
# FCGAN + LP + Embed
checkpoint = torch.load("fcgan_results/wgan_fc_cross_cylinder_TT.pth.tar", map_location=device)
generator2.load_state_dict(checkpoint['generator_state_dict'])
#fig = gan_prediction_plot(generator2, forward_net, test_loader, training_dataset, 60, Z_DIM, Y_DIM)
#fig.savefig(fig_dir + "fcgan_TT_prediction.png", format='png', dpi=100, bbox_inches='tight')
# FCGAN
checkpoint = torch.load("fcgan_results/wgan_fc_cross_cylinder_FF.pth.tar", map_location=device)
generator.load_state_dict(checkpoint['generator_state_dict'])
#fig = gan_prediction_plot(generator, forward_net, test_loader, training_dataset, 60, Z_DIM, Y_DIM)
f#ig.savefig(fig_dir + "fcgan_FF_prediction.png", format='png', dpi=100, bbox_inches='tight')

######### Custom plot from FCGAN and FCGAN + LP + Embed ##########
fig = gan_prediction_plot2(generator, generator2, forward_net, test_loader, training_dataset, 15, Z_DIM, Y_DIM,
                     ["FCGAN", "FCGAN + LP + Embed."])
fig.savefig(fig_dir + "fcgan_predictions.png", format='png', dpi=100, bbox_inches='tight')

fig = gan_prediction_plot3(generator, generator2, forward_net, test_loader, training_dataset, 15, Z_DIM, Y_DIM,
                     ["FCGAN", "FCGAN + LP + Embed."], [0, 4.5e-14])
fig.savefig(fig_dir + "fcgan_predictions_simplified.png", format='png', dpi=150, bbox_inches='tight')
fig.savefig(fig_dir + "fcgan_predictions_simplified.svg", format='svg', dpi=100, bbox_inches='tight')

# DCGAN
checkpoint = torch.load("dcgan_results/wgan_dc_cross_cylinder_FF.pth.tar", map_location=device)
dc_generator.load_state_dict(checkpoint['generator_state_dict'])

# DCGAN + LP + Embed
checkpoint = torch.load("dcgan_results/wgan_dc_cross_cylinder_TT.pth.tar", map_location=device)
dc_generator2.load_state_dict(checkpoint['generator_state_dict'])

fig = gan_prediction_plot3(dc_generator, dc_generator2, forward_net, test_loader, training_dataset, 15, Z_DIM, Y_DIM,
                     ["DCGAN", "DCGAN + LP + Embed."], [0, 4.5e-14])
fig.savefig(fig_dir + "dcgan_predictions_simplified.png", format='png', dpi=150, bbox_inches='tight')
fig.savefig(fig_dir + "dcgan_predictions_simplified.svg", format='svg', dpi=150, bbox_inches='tight')

"""
# Calculate reconstruction error cnn error
mae = estimate_test_reconstruction_error(generator, test_loader, metric=nn.L1Loss(), n=30)
cnn_error = estimate_test_forward_error(generator, test_loader, gan_variable, n=30)
print("Test results")
print(f"Pixelwise MAE: {mae}")
print(f"CNN error: {cnn_error}")
"""