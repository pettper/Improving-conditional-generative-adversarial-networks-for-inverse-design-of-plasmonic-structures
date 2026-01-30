from pathlib import Path

import torch
from matplotlib import pyplot as plt

from src.gan.dcgan.dcgan_generator import DCGANGenerator
from src.gan.fcgan.fc_generator import FullyConnectedGenerator as FCGANGenerator
from src.utils import get_image_size, get_label_size

torch.manual_seed(23)

ZDIM = 100
FEATURE_SCALING = 1


class GANPlotter:
    def __init__(
        self,
        fcgan_checkpoints_dict,
        dcgan_checkpoints_dict,
        train_loader=None,
        val_loader=None,
        device="cpu",
        savefig_dir="./",
    ):
        """
        Expects a dictionary of path strings to stored model checkpoints.
        INPUTS:
            fcgan_checkpoints_dict: A dictionary of 2-tuples, (checkpoint_filename, dropout). The keys will be treated as plot labels.
            dcgan_checkpoints_dict: A dictionary of 2-tuples, (checkpoint_filename, dropout). The keys will be treated as plot labels.
            train_loader: dataloader for the training dataset.
            val_loader: dataloader for the validation dataset.
            device: Torch device to use, defaults to "cpu".
            savefig_dir: Directory to save all figures in.
        """
        self.device = device
        self.savefig_dir = savefig_dir

        # Load checkpoints and dropout rates
        self.fcgan_checkpoints = self._load_checkpoints(fcgan_checkpoints_dict)
        self.dcgan_checkpoints = self._load_checkpoints(dcgan_checkpoints_dict)

        # Data loaders
        self.train_loader = train_loader
        self.val_loader = val_loader

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
                legend_fontsize="xx-small",
            )

        # Upper right, DCGAN validation image MAE
        for k in self.dcgan_checkpoints.keys():
            err = self.dcgan_checkpoints[k]["reconstruction_error"]
            epochs = err[:, 0]
            val_err = err[:, 3]
            self._generic_1d_plot(
                ax[0, 1], epochs, val_err, label=k, legend_fontsize="xx-small"
            )

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
                legend_fontsize="xx-small",
            )

        # Lower right, DCGAN spectral MAE
        for k in self.dcgan_checkpoints.keys():
            err = self.dcgan_checkpoints[k]["forward_error"]
            val_cnn_err = err[:, 4]
            epochs = err[:, 0]
            self._generic_1d_plot(
                ax[1, 1],
                epochs,
                val_cnn_err,
                xlabel="Epoch",
                label=k,
                legend_fontsize="xx-small",
            )

        # Adjust ylim
        for axes in ax.flatten():
            ymin, ymax = axes.get_ylim()
            axes.set_ylim(ymin, ymax * 3.0)

        name = "validation_error_2x2_figure.png"
        path = Path(self.savefig_dir) / Path(name)
        fig.savefig(path, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    def plot_training_and_validation_error(self):
        for k in self.fcgan_checkpoints.keys():
            # Plot image error
            image_error = self.fcgan_checkpoints[k]["reconstruction_error"]
            forward_error = self.fcgan_checkpoints[k]["forward_error"]

            fig = self._train_val_error_plot(
                image_error, forward_error, legend_fontsize="xx-small"
            )
            name = k + "_train_val_error.png"
            path = Path(self.savefig_dir) / Path(name)
            fig.savefig(path, format="png", dpi=150, bbox_inches="tight")
            plt.close(fig)

        for k in self.dcgan_checkpoints.keys():
            # Plot image error
            image_error = self.dcgan_checkpoints[k]["reconstruction_error"]
            forward_error = self.dcgan_checkpoints[k]["forward_error"]

            fig = self._train_val_error_plot(
                image_error, forward_error, legend_fontsize="xx-small"
            )
            name = k + "_train_val_error.png"
            path = Path(self.savefig_dir) / Path(name)
            fig.savefig(path, format="png", dpi=150, bbox_inches="tight")
            plt.close(fig)

    def plot_images(self):
        num_images = 8
        train_x, train_y = next(iter(self.train_loader))
        val_x, val_y = next(iter(self.val_loader))
        z = torch.normal(0, 1, size=(train_x.shape[0], ZDIM)).to(self.device)

        types = ["fc", "dc"]
        for j, cp in enumerate([self.fcgan_checkpoints, self.dcgan_checkpoints]):
            for k in cp.keys():
                generator = self._load_generator(
                    cp[k]["generator_state_dict"],
                    type=types[j],
                    dropout_rate=cp[k]["dropout"],
                )
                generator.eval()

                # Make predictions and plot channel 1
                with torch.no_grad():
                    pred_train_x = generator(z, train_y)
                    pred_val_x = generator(z, val_y)

                fig, axes = plt.subplots(4, num_images, figsize=(18, 10))
                for im in range(num_images):
                    im_train = self._display_on_axis(axes[0, im], train_x[im, 1])
                    _ = self._display_on_axis(axes[1, im], pred_train_x[im, 1])
                    _ = self._display_on_axis(axes[2, im], val_x[im, 1])
                    _ = self._display_on_axis(axes[3, im], pred_val_x[im, 1])

                # Add colorbar using the last image mapped
                cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
                fig.colorbar(im_train, cax=cbar_ax).set_ticks([-1, 0, 1])

                plt.subplots_adjust(right=0.9)

                name = k + "_train_val_pred_images_4x8.png"
                path = Path(self.savefig_dir) / Path(name)
                fig.savefig(path, format="png", dpi=150)
                plt.close(fig)

    def _load_checkpoints(self, checkpoints_dict):
        checkpoints = {}
        for k in checkpoints_dict.keys():
            checkpoints[k] = torch.load(
                checkpoints_dict[k][0], map_location=self.device, weights_only=False
            )
            checkpoints[k]["dropout"] = checkpoints_dict[k][1]
        return checkpoints

    def _load_generator(self, state_dict, type="dc", dropout_rate=0.5):
        im_ch, im_size, _ = get_image_size(self.train_loader)
        lab_ch, ydim = get_label_size(self.train_loader)
        if type == "dc":
            generator = DCGANGenerator(
                im_ch,
                target_channels=lab_ch,
                y_dim=ydim,
                image_size=im_size,
                dropout_rate=dropout_rate,
                features=FEATURE_SCALING,
            )
        elif type == "fc":
            generator = FCGANGenerator(
                im_ch,
                target_channels=lab_ch,
                y_dim=ydim,
                image_size=im_size,
                dropout_rate=dropout_rate,
                features=FEATURE_SCALING,
            )
        else:
            raise TypeError(f"Unknown model type: {type}")
        generator.load_state_dict(state_dict)
        return generator.to(self.device)

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
        legend_fontsize=10,
        legend_loc="upper right",
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
            axes.legend(fontsize=legend_fontsize, loc=legend_loc)

        axes.grid(grid_on, which="both", linewidth=0.5)

    def _train_val_error_plot(
        self,
        image_error,
        forward_error,
        grid_on=True,
        legend_fontsize=10,
        legend_loc="upper right",
    ):
        fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)

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
                legend_fontsize=legend_fontsize,
                legend_loc=legend_loc,
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
                legend_fontsize=legend_fontsize,
                legend_loc=legend_loc,
            )

        return fig

    def _display_on_axis(self, ax, data, cmap="inferno_r"):
        img = data.to(self.device).detach().numpy()
        # Set vmin/vmax to fix colorbar range
        im = ax.imshow(img, cmap=cmap, vmin=-1, vmax=1)
        ax.axis("off")
        return im
