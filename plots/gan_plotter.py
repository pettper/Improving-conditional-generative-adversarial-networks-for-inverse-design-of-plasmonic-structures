from matplotlib import pyplot as plt
import torch
import numpy as np
from pathlib import Path

class GANPlotter:

    def __init__(self, fcgan_checkpoints_dict, dcgan_checkpoints_dict, device='cpu', savefig_dir="./"):
        """
        Expects a dictionary of path strings to stored model checkpoints.
        INPUTS:
            fcgan_checkpoints_dict: A dictionary of strings. The keys will be treated as plot labels.
            dcgan_checkpoints_dict: A dictionary of strings. The keys will be treated as plot labels.
            device: Torch device to use, defaults to "cpu".
        """
        self.device = device
        self.savefig_dir = savefig_dir
        # Load checkpoints
        self.fcgan_checkpoints = self._load_checkpoints(fcgan_checkpoints_dict)
        self.dcgan_checkpoints = self._load_checkpoints(dcgan_checkpoints_dict)

    def plot(self):
        self.plot_error_estimates()

    def plot_error_estimates(self):

        fig, ax = plt.subplots(2,2)

        # Upper left, FCGAN validation image MAE
        for k in self.fcgan_checkpoints.keys():
            err = self.fcgan_checkpoints[k]["reconstruction_error"]
            last_epoch = err[0]
            val_err = err[3]
            epochs = np.arange(1, last_epoch + 1)
            self._generic_1d_plot(ax[0,0], epochs, val_err, xlabel="Epoch", ylabel="$\mathrm{MAE}_{image}$", label=k)

        # Upper right, DCGAN validation image MAE
        for k in self.dcgan_checkpoints.keys():
            err = self.dcgan_checkpoints[k]["reconstruction_error"]
            last_epoch = err[0]
            val_err = err[3]
            epochs = np.arange(1, last_epoch + 1)
            self._generic_1d_plot(ax[0,1], epochs, val_err, xlabel="Epoch", label=k)
        
        # Lower left, FCGAN spectral MAE
        for k in self.fcgan_checkpoints.keys():
            err = self.fcgan_checkpoints[k]["forward_error"]
            last_epoch = err[0]
            val_cnn_err = err[5]
            epochs = np.arange(1, last_epoch + 1)
            self._generic_1d_plot(ax[1,0], epochs, val_cnn_err, xlabel="Epoch", ylabel="$\mathrm{MAE}_{spectra}$", label=k)
        
        # Lower right, DCGAN spectral MAE
        for k in self.dcgan_checkpoints.keys():
            err = self.dcgan_checkpoints[k]["forward_error"]
            last_epoch = err[0]
            val_cnn_err = err[5]
            epochs = np.arange(1, last_epoch + 1)
            self._generic_1d_plot(ax[1,1], epochs, val_cnn_err, xlabel="Epoch", label=k)

        name = "validation_error_2x2_figure"
        path = Path(self.savefig_dir) / Path(name)
        fig.savefig(path, format="png", dpi=150, bbox_inches='tight')

    def plot_training_and_validation_error(self):
        pass

    def _load_checkpoints(self, checkpoints_dict):
        checkpoints = {}
        for k in checkpoints_dict.keys():
            checkpoints[k] = torch.load(checkpoints[k], map_location=self.device)
        return checkpoints

    def _generic_1d_plot(self, axes, xdata, ydata, xlabel=None, ylabel=None, label=None, legend=True, semilogy=True):
        """
        1-dimensional plotting using matplotlib.
        INPUTS:
            axes: plot axes handle
            xdata: x-data to plot
            ydata: y-data to plot
            xlabel: optional x-axis label
            ylabel: optional y-axis label
            label: optional legend label
            legend: use legend True/False
            semilogy: use logarithmic y-axis 
        """
        if semilogy:
            plot_fn = axes.semilogy
        else:
            plot_fn = axes.plot

        plot_fn(xdata, ydata, label=label)
        axes.set_xlabel(xlabel)
        axes.set_ylabel(ylabel)
        axes.legend()

        
