import torch
from matplotlib import pyplot as plt


def gan_prediction_plot(
    generator, forward_network, dataloader, transform_dataset, idx, z_dim, y_dim
):
    real, labels = next(iter(dataloader))
    z = torch.normal(0, 1, size=(labels.shape[0], z_dim))
    fake = generator(z, labels)
    real_labels = forward_network(real)
    fake_labels = forward_network(fake)

    # Inverse training set transform
    dummy_images = torch.ones_like(real)
    _, labels = inverse_training_set_transform(
        dummy_images, labels, transform_dataset, DimerVariable.CROSS_SECTIONS
    )
    _, real_labels = inverse_training_set_transform(
        dummy_images, real_labels, transform_dataset, DimerVariable.CROSS_SECTIONS
    )
    _, fake_labels = inverse_training_set_transform(
        dummy_images, fake_labels, transform_dataset, DimerVariable.CROSS_SECTIONS
    )

    # Prepare for plotting
    labels = labels.detach()
    real_labels = real_labels.detach()
    fake_labels = fake_labels.detach()
    real = real.detach()
    fake = fake.detach()

    original_label = labels[idx]
    real_label = real_labels[idx]
    fake_label = fake_labels[idx]

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
    cmap = plt.get_cmap("inferno")
    cmap = truncate_colormap(cmap, 0.90, 0.0, 500)

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
    ax[2][0].plot(lda, original_label[0], label="FEM", linewidth=3, linestyle="solid")
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
    ax[2][1].plot(lda, original_label[1], label="FEM", linewidth=3, linestyle="solid")
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
                annotations[i][j], xy=(0.1, 0.8), xycoords="axes fraction", fontsize=24
            )

    plt.subplots_adjust(hspace=0.03, wspace=0.33)
    plt.setp(ax[0][0], xticks=[], yticks=[]), plt.setp(ax[0][1], xticks=[], yticks=[])
    plt.setp(ax[1][0], xticks=[], yticks=[]), plt.setp(ax[1][1], xticks=[], yticks=[])

    return fig
