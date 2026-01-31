import torch
from matplotlib import pyplot as plt


def gan_single_prediction_plot(
    generator, forward_network, dataloader, inverse_target_transform, idx, z_dim
):
    generator.eval()
    forward_network.eval()

    with torch.no_grad():
        real, labels = next(iter(dataloader))
        z = torch.normal(0, 1, size=(labels.shape[0], z_dim))
        fake = generator(z, labels)
        real_labels = forward_network(real)
        fake_labels = forward_network(fake)

        # Inverse training set transform
        labels = inverse_target_transform(labels)
        real_labels = inverse_target_transform(real_labels)
        fake_labels = inverse_target_transform(fake_labels)

        # Prepare for plotting
        labels = labels.detach()
        real_labels = real_labels.detach()
        fake_labels = fake_labels.detach()
        real = real.detach()
        fake = fake.detach()

        original_label = labels[idx]
        real_label = real_labels[idx]
        fake_label = fake_labels[idx]
        y_dim = original_label.shape[-1]

        # Custom figure and axes
        fig = plt.figure(figsize=(21, 7))
        ax0 = plt.subplot2grid(shape=(2, 6), loc=(0, 0))
        ax1 = plt.subplot2grid(shape=(2, 6), loc=(0, 1))
        ax2 = plt.subplot2grid(shape=(2, 6), loc=(1, 0))
        ax3 = plt.subplot2grid(shape=(2, 6), loc=(1, 1))
        ax4 = plt.subplot2grid(shape=(2, 6), loc=(0, 2), colspan=2, rowspan=2)
        ax5 = plt.subplot2grid(shape=(2, 6), loc=(0, 4), colspan=2, rowspan=2)
        ax = [[ax0, ax1], [ax2, ax3], [ax4, ax5]]

        # Colormap
        cmap = plt.get_cmap("inferno_r")

        FS = 24
        fs = 18
        (
            ax[0][0].imshow(-real[idx, 0, :, :], vmin=-1, vmax=1, cmap=cmap),
            ax[0][0].set_title("Shape", fontsize=FS),
        )
        (
            ax[0][1].imshow(real[idx, 1, :, :], vmin=-1, vmax=1, cmap=cmap),
            ax[0][1].set_title("Topology", fontsize=FS),
        )
        ax[1][0].imshow(-fake[idx, 0, :, :], vmin=-1, vmax=1, cmap=cmap)
        ax[1][1].imshow(fake[idx, 1, :, :], vmin=-1, vmax=1, cmap=cmap)
        ax[0][0].set_ylabel("Original", fontsize=FS)
        ax[1][0].set_ylabel("GAN", fontsize=FS)
        lda = np.linspace(400, 800, y_dim)
        ax[2][0].plot(
            lda, original_label[0], label="FEM", linewidth=3, linestyle="solid"
        )
        ax[2][0].plot(
            lda, fake_label[0], label="Pred. fake", linewidth=3, linestyle="dashed"
        )
        ax[2][0].plot(
            lda, real_label[0], label="Pred. real", linewidth=3, linestyle="dashdot"
        )
        (
            ax[2][0].set_ylabel("Sca. cross sec. [m^2]", fontsize=FS),
            ax[2][0].legend(fontsize=fs),
        )
        ax[2][0].set_xlabel("Wavelength [nm]", fontsize=FS)
        ax[2][1].plot(
            lda, original_label[1], label="FEM", linewidth=3, linestyle="solid"
        )
        ax[2][1].plot(
            lda, fake_label[1], label="Pred. fake", linewidth=3, linestyle="dashed"
        )
        ax[2][1].plot(
            lda, real_label[1], label="Pred. real", linewidth=3, linestyle="dashdot"
        )
        (
            ax[2][1].set_ylabel("Abs. cross sec. [m^2]", fontsize=FS),
            ax[2][1].legend(fontsize=fs),
        )
        ax[2][1].set_xlabel("Wavelength [nm]", fontsize=FS)
        for i in range(2):
            ax[2][i].yaxis.get_offset_text().set_fontsize(fs)
            ax[2][i].tick_params(axis="y", which="both", labelsize=fs)
            ax[2][i].tick_params(axis="x", which="both", labelsize=fs)
            ax[2][i].set_ylim([0, 5e-14])
            ax[2][i].grid()

        annotations = [["a)", "b)"], ["c)", "d)"], ["e)", "f)"]]
        for i in range(3):
            for j in range(2):
                ax[i][j].annotate(
                    annotations[i][j],
                    xy=(0.1, 0.8),
                    xycoords="axes fraction",
                    fontsize=24,
                )

        plt.subplots_adjust(hspace=0.03, wspace=0.33)
        (
            plt.setp(ax[0][0], xticks=[], yticks=[]),
            plt.setp(ax[0][1], xticks=[], yticks=[]),
        )
        (
            plt.setp(ax[1][0], xticks=[], yticks=[]),
            plt.setp(ax[1][1], xticks=[], yticks=[]),
        )

    return fig


def gan_prediction_comparison(generator_dict, dataloader, indices, z_dim):
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
        ax[0][0].set_ylabel("Original", fontsize=FS, weight="bold")
        for j, k in enumerate(generator_dict.keys()):
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
        _, label = inverse_target_transform(label)
        _, real_labels = inverse_target_transform(real_labels)
        _, fake_labels = inverse_target_transform(fake_labels)

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
        fig, ax = plt.subplots(6, cols, figsize=(12, 10))

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
                )
                ax[(rows // 3) + i][j].plot(
                    lda,
                    real_labels[im, 0, :],
                    label="Pred. real",
                    linewidth=lw,
                    linestyle="dashdot",
                )
                ax[(rows // 3) + i][j].plot(
                    lda,
                    fake_labels[im, 0, :],
                    label="Pred. " + network_label,
                    linewidth=lw,
                    linestyle="dashed",
                )
                # Absorption cross-section
                (line1,) = ax[2 * (rows // 3) + i][j].plot(
                    lda,
                    original_labels[im, 1, :],
                    label="FEM",
                    linewidth=lw,
                    linestyle="solid",
                )
                (line2,) = ax[2 * (rows // 3) + i][j].plot(
                    lda,
                    real_labels[im, 1, :],
                    label="Pred. real",
                    linewidth=lw,
                    linestyle="dashdot",
                )
                (line3,) = ax[2 * (rows // 3) + i][j].plot(
                    lda,
                    fake_labels[im, 1, :],
                    label="Pred. " + network_label,
                    linewidth=lw,
                    linestyle="dashed",
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
                ax[i][j].set_ylim(0, 5.3e-14)
                ax[i][j].grid(True)
                ax[i][j].set_xticks([400, 500, 600, 700, 800])

        # Add a colorbar
        cb_ax = fig.add_axes([0.92, 0.1, 0.02, 0.8])
        cb = fig.colorbar(im_plot, cax=cb_ax)
        cb.ax.tick_params(labelsize=FS + 8)

        plt.subplots_adjust(wspace=0.2, hspace=0.3)
        plt.annotate(
            "Sca. cross sec. [m^2]",
            (0.05, 0.365),
            xycoords="figure fraction",
            fontsize=FS + 6,
            rotation=90,
        )
        plt.annotate(
            "Abs. cross sec. [m^2]",
            (0.05, 0.1),
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
            (0.30, 0.86),
            xycoords="figure fraction",
            fontsize=FS + 8,
            weight="bold",
        )

    return fig
