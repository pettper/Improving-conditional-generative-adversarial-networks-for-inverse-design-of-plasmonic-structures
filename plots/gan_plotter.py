from pathlib import Path

import torch
from matplotlib import pyplot as plt


class GANPlotter:
    def __init__(
        self,
        fcgan_checkpoints_dict,
        dcgan_checkpoints_dict,
        device="cpu",
        savefig_dir="./",
    ):
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
        self.plot_training_and_validation_error()

    def plot_error_estimates(self):
        fig, ax = plt.subplots(2, 2, sharey="row", sharex="col")

        # Upper left, FCGAN validation image MAE
        for k in self.fcgan_checkpoints.keys():
            err = self.fcgan_checkpoints[k]["reconstruction_error"]
            epochs = err[:, 0]
            val_err = err[:, 3]
            self._generic_1d_plot(
                ax[0, 0],
                epochs,
                val_err,
                ylabel=r"$\mathrm{MAE}_{image}$",
                label=k,
            )

        # Upper right, DCGAN validation image MAE
        for k in self.dcgan_checkpoints.keys():
            err = self.dcgan_checkpoints[k]["reconstruction_error"]
            epochs = err[:, 0]
            val_err = err[:, 3]
            self._generic_1d_plot(ax[0, 1], epochs, val_err, label=k)

        # Lower left, FCGAN spectral MAE
        for k in self.fcgan_checkpoints.keys():
            err = self.fcgan_checkpoints[k]["forward_error"]
            val_cnn_err = err[:, 4]
            epochs = err[:, 0]
            self._generic_1d_plot(
                ax[1, 0],
                epochs,
                val_cnn_err,
                xlabel="Epoch",
                ylabel=r"$\mathrm{MAE}_{spectra}$",
                label=k,
            )

        # Lower right, DCGAN spectral MAE
        for k in self.dcgan_checkpoints.keys():
            err = self.dcgan_checkpoints[k]["forward_error"]
            val_cnn_err = err[:, 4]
            epochs = err[:, 0]
            self._generic_1d_plot(
                ax[1, 1], epochs, val_cnn_err, xlabel="Epoch", label=k
            )

        name = "validation_error_2x2_figure.png"
        path = Path(self.savefig_dir) / Path(name)
        fig.savefig(path, format="png", dpi=150, bbox_inches="tight")

    def plot_training_and_validation_error(self):
        for k in self.fcgan_checkpoints.keys():
            # Plot image error
            image_error = self.fcgan_checkpoints[k]["reconstruction_error"]
            forward_error = self.fcgan_checkpoints[k]["forward_error"]

            fig = self._train_val_error_plot(image_error, forward_error)
            name = k + "_train_val_error.png"
            path = Path(self.savefig_dir) / Path(name)
            fig.savefig(path, format="png", dpi=150, bbox_inches="tight")

        for k in self.dcgan_checkpoints.keys():
            # Plot image error
            image_error = self.dcgan_checkpoints[k]["reconstruction_error"]
            forward_error = self.dcgan_checkpoints[k]["forward_error"]

            fig = self._train_val_error_plot(image_error, forward_error)
            name = k + "_train_val_error.png"
            path = Path(self.savefig_dir) / Path(name)
            fig.savefig(path, format="png", dpi=150, bbox_inches="tight")

    def _load_checkpoints(self, checkpoints_dict):
        checkpoints = {}
        for k in checkpoints_dict.keys():
            checkpoints[k] = torch.load(
                checkpoints_dict[k], map_location=self.device, weights_only=False
            )
        return checkpoints

    def _generic_1d_plot(
        self,
        axes,
        xdata,
        ydata,
        xlabel=None,
        ylabel=None,
        label=None,
        legend=True,
        semilogy=True,
        grid_on=True,
    ):
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
        if legend:
            axes.legend()

        axes.grid(grid_on, which="both", linewidth=0.5)

    def _train_val_error_plot(self, image_error, forward_error, grid_on=True):
        fig, ax = plt.subplots(1, 2, sharex=True)

        # Plot image error
        epochs = image_error[:, 0]
        for j, plot_label in enumerate(
            [
                "training data",
                "validation data",
                "struct training data",
                "struct validation data",
            ]
        ):
            c = 2 * j + 1  # c=1,3,5,7
            err = image_error[:, c]
            self._generic_1d_plot(
                ax[0],
                epochs,
                err,
                xlabel="Epochs",
                ylabel=r"$\mathrm{MAE}_{image}$",
                label=plot_label,
                grid_on=grid_on,
            )

        # Plot forward error
        epochs = forward_error[:, 0]

        for j, plot_label in enumerate(["training data", "validation data"]):
            c = 2 * (j + 1)  # c=2,4
            err = forward_error[:, c]
            self._generic_1d_plot(
                ax[1],
                epochs,
                err,
                xlabel="Epochs",
                ylabel=r"$\mathrm{MAE}_{spectra}$",
                label=plot_label,
                grid_on=grid_on,
            )

        return fig
