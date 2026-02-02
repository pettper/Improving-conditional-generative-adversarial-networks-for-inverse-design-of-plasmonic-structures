import numpy as np
import torch
from matplotlib import pyplot as plt


def gan_single_prediction_plot(
    generator1,
    generator2,
    forward_network,
    dataloader,
    inverse_target_transform,
    idx,
    z_dim,
    network_labels,
):
    generator1.eval()
    generator2.eval()
    forward_network.eval()

    with torch.no_grad():
        real, label = next(iter(dataloader))
        real, label = (real[idx].unsqueeze(0), label[idx].unsqueeze(0))

        z = torch.normal(0, 1, size=(label.shape[0], z_dim))
        fake = generator1(z, label)
        fake2 = generator2(z, label)
        real_label = forward_network(real)
        fake_label = forward_network(fake)
        fake_label2 = forward_network(fake2)

        # Inverse target transform to restore original data range
        label = inverse_target_transform(label)
        real_label = inverse_target_transform(real_label)
        fake_label = inverse_target_transform(fake_label)
        fake_label2 = inverse_target_transform(fake_label2)

        # Prepare for plotting
        label = label.detach()
        real_label = real_label.detach()
        fake_label = fake_label.detach()
        fake_label2 = fake_label2.detach()
        real = real.detach()
        fake = fake.detach()
        fake2 = fake2.detach()

        original_label = label[0]
        ydim = original_label.shape[-1]
        ylim = 1.75 * original_label.max().item()
        real_label = real_label[0]
        fake_label = fake_label[0]
        fake_label2 = fake_label2[0]
        real = real[0]
        fake = fake[0]
        fake2 = fake2[0]

        # Custom figure and axes
        fig = plt.figure(figsize=(16, 10))
        ax0 = plt.subplot2grid(shape=(5, 6), loc=(0, 0), colspan=2, rowspan=2)
        ax1 = plt.subplot2grid(shape=(5, 6), loc=(0, 2), colspan=2, rowspan=2)
        ax2 = plt.subplot2grid(shape=(5, 6), loc=(0, 4), colspan=2, rowspan=2)
        ax3 = plt.subplot2grid(shape=(5, 6), loc=(2, 0), colspan=3, rowspan=3)
        ax4 = plt.subplot2grid(
            shape=(5, 6), loc=(2, 3), colspan=3, rowspan=3, sharey=ax3
        )
        ax = [ax0, ax1, ax2, ax3, ax4]

        # Colormap
        cmap = plt.get_cmap("inferno_r")

        FS = 16
        fs = 14
        # Original image plot
        im_plot = ax0.imshow(real[1, :, :], vmin=-1, vmax=1, cmap=cmap)
        ax0.set_title("Original", fontsize=FS, weight="bold")
        # GAN-network 1 image plot
        ax1.imshow(fake[1, :, :], vmin=-1, vmax=1, cmap=cmap)
        ax1.set_title(network_labels[0], fontsize=FS, weight="bold")
        # GAN-network 2 image plot
        ax2.imshow(fake2[1, :, :], vmin=-1, vmax=1, cmap=cmap)
        ax2.set_title(network_labels[1], fontsize=FS, weight="bold")
        # Prediction plots, GAN-network 1
        lda = np.linspace(400, 800, ydim)
        ax3.plot(lda, original_label[0], label="FEM", linewidth=3, linestyle="solid")
        ax3.plot(
            lda,
            fake_label[0],
            label="Pred. " + network_labels[0],
            linewidth=3,
            linestyle="dashed",
        )
        ax3.plot(
            lda,
            fake_label2[0],
            label="Pred. " + network_labels[1],
            linewidth=3,
            linestyle="dashed",
        )
        ax3.plot(
            lda, real_label[0], label="Pred. real", linewidth=3, linestyle="dashdot"
        )
        ax3.set_xlabel("Wavelength [nm]", fontsize=FS)
        ax3.set_ylabel("Sca. cross sec. [m^2]", fontsize=FS)
        ax3.legend(fontsize=fs)
        ax4.plot(lda, original_label[1], label="FEM", linewidth=3, linestyle="solid")
        ax4.plot(
            lda,
            fake_label[1],
            label="Pred. " + network_labels[0],
            linewidth=3,
            linestyle="dashed",
        )
        ax4.plot(
            lda,
            fake_label2[1],
            label="Pred. " + network_labels[1],
            linewidth=3,
            linestyle="dashed",
        )
        ax4.plot(
            lda, real_label[1], label="Pred. real", linewidth=3, linestyle="dashdot"
        )
        ax4.set_xlabel("Wavelength [nm]", fontsize=FS)
        ax4.set_ylabel("Abs. cross sec. [m^2]", fontsize=FS)
        ax4.legend(fontsize=fs)

        # Adjust x-axis and y-axis
        for j in range(3, 5):
            ax[j].yaxis.get_offset_text().set_fontsize(fs)
            ax[j].tick_params(axis="y", which="both", labelsize=fs)
            ax[j].tick_params(axis="x", which="both", labelsize=fs)
            ax[j].set_ylim(0.0, ylim)
            ax[j].grid()

        annotations = ["a)", "b)", "c)", "d)", "e)"]
        for i in range(5):
            ax[i].annotate(
                annotations[i], xy=(0.1, 0.8), xycoords="axes fraction", fontsize=FS
            )

        cb_ax = fig.add_axes([0.92, 0.1, 0.02, 0.8])
        cb = fig.colorbar(im_plot, cax=cb_ax)
        cb.ax.tick_params(labelsize=FS)

        # Adjust space between subplots
        plt.subplots_adjust(hspace=0.2, wspace=0.65)

        # Remove ticks on image plots
        for i in range(3):
            plt.setp(ax[i], xticks=[], yticks=[])
        return fig


def gan_prediction_comparison(
    generator_dict, dataloader, indices, z_dim, generator_labels=None
):
    num_images = 6
    assert len(indices) == num_images
    number_of_generators = len(generator_dict.keys())

    with torch.no_grad():
        # Get data
        real, label = next(iter(dataloader))
        real, label = (real[indices], label[indices])
        z = torch.normal(0, 1, size=(label.shape[0], z_dim))
        real = real.detach()

        # Make a subplots grid to plot samples from the generators + original image
        fig, ax = plt.subplots(
            number_of_generators + 1,
            num_images,
            figsize=(12, 2 * (number_of_generators + 1)),
        )

        # Colormap
        cmap = plt.get_cmap("inferno_r")

        # Font sizes
        FS = 14
        fs = 6

        # Plot images
        for i in range(num_images):
            im_plot = ax[0][i].imshow(
                real[i, 1, :, :], vmin=-1, vmax=1, cmap=cmap
            )  # To plot the original images
            plt.setp(
                ax[0][i], xticks=[], yticks=[]
            )  # To remove the ticks from all images

        for j, k in enumerate(generator_dict.keys()):
            generator = generator_dict[k]
            generator.eval()
            fake = generator(z, label).detach()
            for i in range(num_images):
                ax[j + 1][i].imshow(
                    fake[i, 1, :, :], vmin=-1, vmax=1, cmap=cmap
                )  # To plot images from GAN-network
                plt.setp(
                    ax[j + 1][i], xticks=[], yticks=[]
                )  # To remove the ticks from all images

        # Adds network labels
        if generator_labels:
            network_labels = generator_labels
        else:
            network_labels = generator_dict.keys()
        ax[0][0].set_ylabel("Original", fontsize=FS, weight="bold")
        for j, k in enumerate(network_labels):
            ax[j + 1][0].set_ylabel(k, fontsize=FS, weight="bold")

        # Adds a colorbar
        cb_ax = fig.add_axes([0.92, 0.1, 0.02, 0.8])
        cb = fig.colorbar(im_plot, cax=cb_ax)
        cb.ax.tick_params(labelsize=FS)

    return fig


def gan_big_prediction_plot(
    generator,
    forward_network,
    dataloader,
    inverse_target_transform,
    indices,
    z_dim,
    network_label,
):
    generator.eval()
    forward_network.eval()
    with torch.no_grad():
        # Get data
        real, label = next(iter(dataloader))
        real, label = (real[indices], label[indices])

        # Generate images
        z = torch.normal(0, 1, size=(label.shape[0], z_dim))
        fake = generator(z, label)
        real_labels = forward_network(real)
        fake_labels = forward_network(fake)

        # Apply inverse transform to the labels to retain original data range
        label = inverse_target_transform(label)
        real_labels = inverse_target_transform(real_labels)
        fake_labels = inverse_target_transform(fake_labels)

        # Prepare for plotting
        original_labels = label.detach()
        real_labels = real_labels.detach()
        fake_labels = fake_labels.detach()
        y_dim = original_labels.shape[-1]
        real = real.detach()
        fake = fake.detach()

        # 6x5 subplot
        rows = 6
        cols = 5
        fig, ax = plt.subplots(6, cols, figsize=(13, 10))

        # Colormap
        cmap = plt.get_cmap("inferno_r")

        # Font size
        FS = 8

        # Plot images and labels
        lda = np.linspace(400, 800, y_dim)  # Wavelength
        lw = 1.5  # line width
        for i in range(rows // 3):
            for j in range(cols):
                im = i * cols + j
                im_plot = ax[i][j].imshow(
                    torch.cat((real[im, 1, :, :], fake[im, 1, :, :]), dim=1),
                    vmin=-1,
                    vmax=1,
                    cmap=cmap,
                )  # Original images plot
                ax[i][j].axvline(
                    128, color="black", linewidth=1
                )  # Split images with a vertical line
                # Scattering cross-section
                ax[(rows // 3) + i][j].plot(
                    lda,
                    original_labels[im, 0, :],
                    label="FEM",
                    linewidth=lw,
                    linestyle="solid",
                    marker="",
                )
                ax[(rows // 3) + i][j].plot(
                    lda,
                    real_labels[im, 0, :],
                    label="Pred. real",
                    linewidth=lw,
                    linestyle="dashdot",
                    marker="",
                )
                ax[(rows // 3) + i][j].plot(
                    lda,
                    fake_labels[im, 0, :],
                    label="Pred. " + network_label,
                    linewidth=lw,
                    linestyle="dashed",
                    marker="",
                )
                # Absorption cross-section
                (line1,) = ax[2 * (rows // 3) + i][j].plot(
                    lda,
                    original_labels[im, 1, :],
                    label="FEM",
                    linewidth=lw,
                    linestyle="solid",
                    marker="",
                )
                (line2,) = ax[2 * (rows // 3) + i][j].plot(
                    lda,
                    real_labels[im, 1, :],
                    label="Pred. real",
                    linewidth=lw,
                    linestyle="dashdot",
                    marker="",
                )
                (line3,) = ax[2 * (rows // 3) + i][j].plot(
                    lda,
                    fake_labels[im, 1, :],
                    label="Pred. " + network_label,
                    linewidth=lw,
                    linestyle="dashed",
                    marker="",
                )
                plt.figlegend(
                    handles=[line1, line2, line3],
                    fontsize=FS + 4,
                    loc="lower right",
                    ncol=3,
                    bbox_to_anchor=(1.00, 0.02),
                )

        annotations = [
            ["a)", "b)", "c)", "d)", "e)", "f)"],
            ["g)", "h)", "i)", "j)", "k)", "l)"],
            ["m)", "n)", "o)", "p)", "q)", "r)"],
        ]
        # Remove ticks on image plots and ticklabels label plot
        for i in range(rows // 3):
            for j in range(cols):
                plt.setp(ax[i][j], xticks=[], yticks=[])
                plt.setp(ax[(rows // 3) + i][j], xticklabels=[])
                if i < 1:
                    plt.setp(ax[2 * (rows // 3) + i][j], xticklabels=[])
                ax[i][j].annotate(
                    annotations[i][j],
                    fontsize=FS + 4,
                    xy=(0.08, 0.80),
                    xycoords="axes fraction",
                )
                ax[(rows // 3) + i][j].annotate(
                    annotations[i][j],
                    fontsize=FS + 4,
                    xy=(0.08, 0.80),
                    xycoords="axes fraction",
                )
                ax[2 * (rows // 3) + i][j].annotate(
                    annotations[i][j],
                    fontsize=FS + 4,
                    xy=(0.08, 0.80),
                    xycoords="axes fraction",
                )

        # Set axes limits and grid to true for cross section plots
        for i in range(rows // 3, rows):
            for j in range(cols):
                ax[i][j].set_ylim(0, 6.0e-14)
                ax[i][j].grid(True)
                ax[i][j].set_xticks([400, 500, 600, 700, 800])

        # Add a colorbar
        cb_ax = fig.add_axes([0.92, 0.1, 0.02, 0.8])
        cb = fig.colorbar(im_plot, cax=cb_ax)
        cb.ax.tick_params(labelsize=FS + 8)

        plt.subplots_adjust(wspace=0.2, hspace=0.3)
        plt.annotate(
            "Sca. cross sec. [m^2]",
            (0.08, 0.385),
            xycoords="figure fraction",
            fontsize=FS + 6,
            rotation=90,
        )
        plt.annotate(
            "Abs. cross sec. [m^2]",
            (0.08, 0.12),
            xycoords="figure fraction",
            fontsize=FS + 6,
            rotation=90,
        )
        plt.annotate(
            "Wavelength [nm]",
            (0.403, 0.05),
            xycoords="figure fraction",
            fontsize=FS + 6,
        )
        plt.annotate(
            "Original | " + network_label,
            (0.35, 0.90),
            xycoords="figure fraction",
            fontsize=FS + 8,
            weight="bold",
        )

    return fig
