from pathlib import Path

import numpy as np
import torch
from cycler import cycler
from matplotlib import pyplot as plt
from torch.nn import Softplus
from torch.utils.data import DataLoader, RandomSampler

from plots.plot_help_functions import (
    data_samples_plot,
    gan_big_prediction_plot,
    gan_prediction_comparison,
    gan_single_prediction_plot,
    gaussian_spectrum_plot,
)
from src.gan.dcgan.dcgan_generator import DCGANGenerator
from src.gan.fcgan.fc_generator import FullyConnectedGenerator as FCGANGenerator
from src.utils import gaussian_fun, get_image_size, get_label_size, moving_average

torch.manual_seed(23)

ZDIM = 100
SMA_WINDOW_SIZE = 3
DPI = 250

colors = ["#003049", "#D62828", "#F77F00", "#FCBF49", "#EAE2B7", "#588157"]
markers = ["o", "s", "^", "D", "v", "p"]
plt.rcParams["axes.prop_cycle"] = cycler(color=colors) + cycler(marker=markers)
plt.rcParams["lines.markersize"] = 3  # Smaller, more subtle markers
plt.rcParams["lines.markerfacecolor"] = "none"
plt.rcParams["lines.markeredgewidth"] = 0.5  # Keeps the marker border thin


class GANPlotter:
    def __init__(
        self,
        fcgan_checkpoints_dict,
        dcgan_checkpoints_dict,
        forward_network,
        train_dataset,
        val_dataset,
        test_dataset,
        fcgan_feature_scaling=1,
        dcgan_feature_scaling=1,
        device="cpu",
        savefig_dir="./",
    ):
        """
        Expects a dictionary of path strings to stored model checkpoints.
        INPUTS:
            fcgan_checkpoints_dict: A dictionary of 2-tuples, (checkpoint_filename, dropout). The keys will be treated as plot labels.
            dcgan_checkpoints_dict: A dictionary of 2-tuples, (checkpoint_filename, dropout). The keys will be treated as plot labels.
            train_dataset: Training dataset, an instance of DimerDataset.
            val_dataset: Validation dataset, an instance of DimerDataset.
            device: Torch device to use, defaults to "cpu".
            savefig_dir: Directory to save all figures in.
        """
        self.device = device
        self.savefig_dir = savefig_dir

        # Load checkpoints and dropout rates
        self.fcgan_checkpoints = self._load_checkpoints(fcgan_checkpoints_dict)
        self.dcgan_checkpoints = self._load_checkpoints(dcgan_checkpoints_dict)

        # Dataset and data loaders
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.train_loader = GANPlotter._setup_data_loader(
            self.train_dataset, RandomSampler(self.train_dataset)
        )
        self.val_loader = GANPlotter._setup_data_loader(
            self.val_dataset, RandomSampler(self.val_dataset)
        )
        self.test_loader = GANPlotter._setup_data_loader(self.test_dataset)

        # Forward network is used for evaluation
        self.forward_network = self._load_forward_network(forward_network)
        self.forward_network.eval()

        # Feature scaling
        self.fc_features = fcgan_feature_scaling
        self.dc_features = dcgan_feature_scaling

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
                legend_fontsize="x-small",
                apply_sma_smoothing=True,
                markevery=3,
            )

        # Upper right, DCGAN validation image MAE
        for k in self.dcgan_checkpoints.keys():
            err = self.dcgan_checkpoints[k]["reconstruction_error"]
            epochs = err[:, 0]
            val_err = err[:, 3]
            self._generic_1d_plot(
                ax[0, 1],
                epochs,
                val_err,
                label=k,
                legend_fontsize="x-small",
                apply_sma_smoothing=True,
                markevery=3,
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
                legend_fontsize="x-small",
                apply_sma_smoothing=True,
                markevery=3,
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
                legend_fontsize="x-small",
                apply_sma_smoothing=True,
                markevery=3,
            )

        # Adjust ylim
        for axes in ax.flatten():
            ymin, ymax = axes.get_ylim()
            axes.set_ylim(0.8 * ymin, ymax * 1.75)

        # To label each subfigure
        labels = ["a)", "b)", "c)", "d)"]
        for i, axes in enumerate(ax.flatten()):
            axes.text(
                0.13,
                0.91,
                labels[i],
                transform=axes.transAxes,
                fontsize="medium",
                va="top",
                ha="right",
            )

        name = "validation_error_2x2_figure"
        path = Path(self.savefig_dir) / Path(name)
        fig.savefig(str(path) + ".png", format="png", dpi=DPI, bbox_inches="tight")
        fig.savefig(str(path) + ".svg", format="svg", dpi=DPI, bbox_inches="tight")
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
            fig.savefig(path, format="png", dpi=DPI, bbox_inches="tight")
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
            fig.savefig(path, format="png", dpi=DPI, bbox_inches="tight")
            plt.close(fig)

    def plot_images(self):
        num_images = 8
        train_x, train_y = next(iter(self.train_loader))
        val_x, val_y = next(iter(self.val_loader))
        train_z = torch.normal(0, 1, size=(train_x.shape[0], ZDIM)).to(self.device)
        val_z = torch.normal(0, 1, size=(val_x.shape[0], ZDIM)).to(self.device)

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
                    pred_train_x = generator(train_z, train_y)
                    pred_val_x = generator(val_z, val_y)

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

                axes[0, 0].set_ylabel("Traning data")
                axes[1, 0].set_ylabel("Train. prediction")
                axes[2, 0].set_ylabel("Validation data")
                axes[3, 0].set_ylabel("Val. prediction")

                name = k + "_train_val_pred_images_4x8.png"
                path = Path(self.savefig_dir) / Path(name)
                fig.savefig(path, format="png", dpi=DPI, bbox_inches="tight")
                plt.close(fig)

    def plot_single_sample_prediction(self, fcgan_keys=None, dcgan_keys=None):
        """
        Expects at least one of fcgan_keys or dcgan_keys to be a 2-tuple of keys to use in plot.
        """
        idx = 38
        configs = (
            ("fc", fcgan_keys, self.fcgan_checkpoints),
            ("dc", dcgan_keys, self.dcgan_checkpoints),
        )

        for type, keys, checkpoints in configs:
            if keys:
                assert len(keys) == 2

                generators = []
                generator_labels = []
                for k in keys:
                    generators.append(
                        self._load_generator(
                            checkpoints[k]["generator_state_dict"],
                            type=type,
                            dropout_rate=checkpoints[k]["dropout"],
                        )
                    )
                    generator_labels.append(k)

                fig = gan_single_prediction_plot(
                    generators[0],
                    generators[1],
                    self.forward_network,
                    self.test_loader,
                    lambda x: self.test_dataset.apply_inverse_target_transform(x),
                    idx,
                    ZDIM,
                    generator_labels,
                )

                name = type + "gan_single_prediction"
                path = Path(self.savefig_dir) / Path(name)
                fig.savefig(
                    str(path) + ".png", format="png", dpi=DPI, bbox_inches="tight"
                )
                fig.savefig(
                    str(path) + ".svg", format="svg", dpi=DPI, bbox_inches="tight"
                )
                plt.close(fig)

    def plot_prediction_comparison(
        self, gan_keys, figname="gan_model_prediction_comparison"
    ):
        six_indices = [110, 4, 301, 256, 155, 406]
        types = ["fc", "dc"]
        network_labels = []
        generator_dict = {}
        for j, cp in enumerate([self.fcgan_checkpoints, self.dcgan_checkpoints]):
            for k in cp.keys():
                if k in gan_keys:
                    generator = self._load_generator(
                        cp[k]["generator_state_dict"],
                        type=types[j],
                        dropout_rate=cp[k]["dropout"],
                    )
                    generator_dict[k] = generator
                    network_labels.append(k)
        fig = gan_prediction_comparison(
            generator_dict,
            self.test_loader,
            six_indices,
            ZDIM,
            generator_labels=network_labels,
        )

        path = Path(self.savefig_dir) / Path(figname)
        fig.savefig(str(path) + ".png", format="png", dpi=DPI, bbox_inches="tight")
        fig.savefig(str(path) + ".eps", format="eps", dpi=DPI, bbox_inches="tight")
        plt.close(fig)

    def plot_multiple_predictions(self, gan_key_pair, indices=None, name_suffix=""):
        assert isinstance(gan_key_pair, tuple) and len(gan_key_pair) == 2
        if indices:
            idx = indices
        else:
            idx = [6, 18, 48, 59, 143, 224, 255, 285, 298, 426]
            idx = [255, 48, 426, 78, 143, 18, 59, 298]  # 6
        generators = []
        for k in gan_key_pair:
            if k in self.fcgan_checkpoints.keys():
                cp = self.fcgan_checkpoints
                g = self._load_generator(
                    cp[k]["generator_state_dict"],
                    type="fc",
                    dropout_rate=cp[k]["dropout"],
                )
                generators.append(g)
            elif k in self.dcgan_checkpoints.keys():
                cp = self.dcgan_checkpoints
                g = self._load_generator(
                    cp[k]["generator_state_dict"],
                    type="dc",
                    dropout_rate=cp[k]["dropout"],
                )
                generators.append(g)
            else:
                raise ValueError(
                    f"Provided key {k} is not found in any checkpoint dict."
                )
        fig = gan_big_prediction_plot(
            generators,
            self.forward_network,
            self.test_loader,
            lambda x: self.test_dataset.apply_inverse_target_transform(x),
            idx,
            ZDIM,
            gan_key_pair,
        )

        assert type(name_suffix) is str
        name = "multiple_predictions_" + name_suffix
        path = Path(self.savefig_dir) / Path(name)
        fig.savefig(
            str(path) + ".png",
            format="png",
            dpi=DPI,
        )
        fig.savefig(
            str(path) + ".eps",
            format="eps",
            dpi=DPI,
        )
        plt.close(fig)

    def plot_gaussian_predictions(self, fcgan_key_pair=None, dcgan_key_pair=None):
        master_fig = plt.figure(figsize=(16, 12))
        subfigs = master_fig.subfigures(3, 1)

        with torch.no_grad():
            lab_ch, ydim = get_label_size(self.train_loader)
            lda = np.linspace(400, 800, ydim)
            sca = np.stack(
                [
                    gaussian_fun(lda, 0.4e-14, 650, 35),
                    gaussian_fun(lda, 3.2e-14, 650, 40),
                    gaussian_fun(lda, 1.3e-14, 610, 35),
                ]
            )

            abs = np.stack(
                [
                    gaussian_fun(lda, 2.0e-14, 610, 35),
                    gaussian_fun(lda, 2.9e-14, 575, 60),
                    gaussian_fun(lda, 2.0e-14, 610, 55),
                ]
            )
            y = torch.Tensor(np.stack([sca, abs], axis=1))
            y = self.train_dataset.apply_target_transform(y)
            z = torch.normal(0, 1, size=(3, ZDIM))

            if fcgan_key_pair:
                x_list = []
                y_pred_list = []
                labels = []
                for k in fcgan_key_pair:
                    generator = self._load_generator(
                        self.fcgan_checkpoints[k]["generator_state_dict"],
                        type="fc",
                        dropout_rate=self.fcgan_checkpoints[k]["dropout"],
                    )
                    generator.eval()
                    x = generator(z, y)
                    y_pred = self.train_dataset.apply_inverse_target_transform(
                        self.forward_network(x)
                    )
                    x_list.append(x)
                    y_pred_list.append(y_pred)
                    labels.append(k)

                for j in range(len(subfigs)):
                    gaussian_spectrum_plot(
                        subfigs[j],
                        [x_list[0][j, :, :, :], x_list[1][j, :, :, :]],
                        [y_pred_list[0][j], y_pred_list[1][j]],
                        labels,
                        lda,
                        sca[j],
                        abs[j],
                    )

                labels = ["a", "b", "c", "d", "e", "f", "g", "h", "i"]
                x_pos = np.array([0.08, 0.29, 0.51]) + 0.07
                y_pos = np.array([0.96, 0.63, 0.30]) - 0.045

                for j, label in enumerate(labels):
                    master_fig.text(
                        x_pos[j % 3],
                        y_pos[j // 3],
                        f"({label})",
                        fontsize=12,
                    )

                # To label the image columns with network labels
                ax = subfigs[0].get_axes()
                ax[1].set_title(f"{fcgan_key_pair[0]}", fontsize=14, fontweight="bold")
                ax[2].set_title(f"{fcgan_key_pair[1]}", fontsize=14, fontweight="bold")

                name = "fc_gaussian"
                path = Path(self.savefig_dir) / Path(name)
                master_fig.savefig(
                    str(path) + ".png", format="png", dpi=DPI, bbox_inches="tight"
                )
                master_fig.savefig(
                    str(path) + ".eps", format="eps", dpi=DPI, bbox_inches="tight"
                )
                plt.close(master_fig)

            if dcgan_key_pair:
                x_list = []
                y_pred_list = []
                labels = []
                for k in dcgan_key_pair:
                    generator = self._load_generator(
                        self.dcgan_checkpoints[k]["generator_state_dict"],
                        type="dc",
                        dropout_rate=self.dcgan_checkpoints[k]["dropout"],
                    )
                    generator.eval()
                    x = generator(z, y)
                    y_pred = self.train_dataset.apply_inverse_target_transform(
                        self.forward_network(x)
                    )
                    x_list.append(x)
                    y_pred_list.append(y_pred)
                    labels.append(k)

                for j in range(len(subfigs)):
                    gaussian_spectrum_plot(
                        subfigs[j],
                        [x_list[0][j, :, :, :], x_list[1][j, :, :, :]],
                        [y_pred_list[0][j], y_pred_list[1][j]],
                        labels,
                        lda,
                        sca[j],
                        abs[j],
                    )

                labels = ["a", "b", "c", "d", "e", "f", "g", "h", "i"]
                x_pos = np.array([0.08, 0.29, 0.51]) + 0.07
                y_pos = np.array([0.96, 0.63, 0.30]) - 0.045

                for j, label in enumerate(labels):
                    master_fig.text(
                        x_pos[j % 3],
                        y_pos[j // 3],
                        f"({label})",
                        fontsize=12,
                    )

                # To label the image columns with network labels
                ax = subfigs[0].get_axes()
                ax[1].set_title(f"{dcgan_key_pair[0]}", fontsize=14, fontweight="bold")
                ax[2].set_title(f"{dcgan_key_pair[1]}", fontsize=14, fontweight="bold")

                name = "dc_gaussian"
                path = Path(self.savefig_dir) / Path(name)
                master_fig.savefig(
                    str(path) + ".png", format="png", dpi=DPI, bbox_inches="tight"
                )
                master_fig.savefig(
                    str(path) + ".eps", format="eps", dpi=DPI, bbox_inches="tight"
                )
                plt.close(master_fig)

    def plot_data_samples(self, prefix="nano"):
        fig = data_samples_plot(
            self.train_loader,
            lambda x: self.train_dataset.apply_inverse_target_transform(x),
        )
        name = prefix + "_data"
        path = Path(self.savefig_dir) / Path(name)
        fig.savefig(str(path) + ".png", format="png", dpi=DPI, bbox_inches="tight")
        fig.savefig(str(path) + ".eps", format="eps", dpi=DPI, bbox_inches="tight")
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
                features=self.dc_features,
            )
        elif type == "fc":
            generator = FCGANGenerator(
                im_ch,
                target_channels=lab_ch,
                y_dim=ydim,
                image_size=im_size,
                dropout_rate=dropout_rate,
                features=self.fc_features,
            )
        else:
            raise TypeError(f"Unknown model type: {type}")
        generator.load_state_dict(state_dict)
        return generator.to(self.device)

    def _load_forward_network(self, forward_network_dict):
        im_ch, im_size, _ = get_image_size(self.train_loader)
        out_ch, ydim = get_label_size(self.train_loader)

        fn_checkpoint = torch.load(
            forward_network_dict["load_path"],
            weights_only=False,
        )
        fn = forward_network_dict["model"]
        forward_network = fn(
            im_ch,
            ydim,
            activation=Softplus(),
            image_size=im_size,
            out_channels=out_ch,
            dropout_rate=0.5,
        )
        forward_network.load_state_dict(fn_checkpoint["model_state_dict"])
        return forward_network.to(self.device)

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
        apply_sma_smoothing=False,
        linewidth=1,
        markevery=1,
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

        if apply_sma_smoothing:
            ydata = moving_average(ydata, SMA_WINDOW_SIZE)

        plot_fn(xdata, ydata, label=label, linewidth=linewidth, markevery=markevery)
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
                apply_sma_smoothing=True,
                markevery=3,
                linewidth=1,
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
                apply_sma_smoothing=True,
                markevery=3,
                linewidth=1,
            )

        return fig

    def _display_on_axis(self, ax, data, cmap="inferno_r"):
        img = data.to(self.device).detach().numpy()
        # Set vmin/vmax to fix colorbar range
        im = ax.imshow(img, cmap=cmap, vmin=-1, vmax=1)
        ax.axis("off")
        return im

    @staticmethod
    def _setup_data_loader(dataset, sampler=None):
        loader = None
        if dataset:
            loader = DataLoader(
                dataset,
                batch_size=3000,
                sampler=sampler,
                pin_memory=True,
            )
        return loader
