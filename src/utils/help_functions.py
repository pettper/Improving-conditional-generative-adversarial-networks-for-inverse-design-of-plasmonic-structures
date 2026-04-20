import re
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.stats import norm
from torch import autograd, nn
from torch.func import jacfwd, jacrev, vmap
from torch.linalg import matrix_norm


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def conv_output_size(input_size, kernel, padding, stride):
    # Assumes square input and kernel size
    return (input_size - kernel + 2 * padding) // stride + 1


def initialize_dcgan_weights(model):
    # Initializes weights according to the dcgan paper
    for m in model.modules():
        if isinstance(
            m,
            (
                nn.Conv2d,
                nn.ConvTranspose2d,
                nn.BatchNorm2d,
                nn.InstanceNorm2d,
                nn.Linear,
            ),
        ):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


def clear_old_files(filepath, extensions=".pth.tar"):
    """
    Clear old files with a regex pattern matching the filename provided in 'filepath'
    """
    filepath = Path(filepath)
    clean_name = filepath.name.removesuffix(extensions)
    pattern = re.compile(rf"{re.escape(clean_name)}.*")

    search_dir = filepath.parent

    for file in search_dir.glob("*.pth.tar"):
        if file.is_file() and pattern.search(file.name):
            print(f"Removing file: {file.name}")
            file.unlink()


def add_filename_suffix(filepath, suffix, extensions=".pth.tar"):
    """
    Adds a suffix to a filename while preserving directory paths
    and multi-part extensions (e.g., .pth.tar).
    """
    path = Path(filepath)

    # path.parent is the directory (e.g., 'checkpoints/models')
    # path.name.removesuffix(extensions) gets the base name (e.g., 'wgan_dc_cross_all_TT')

    base_name = path.name.removesuffix(extensions)

    # Combine back into a new Path object
    new_path = path.parent / f"{base_name}{suffix}{extensions}"

    return str(new_path)


def load_yaml(settings_path):
    # Function to load settings from yaml-file
    try:
        with open(settings_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            if data is None:
                return {}
            return data
    except FileNotFoundError:
        raise FileNotFoundError(f"YAML file not found: {settings_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in {settings_path}: {e}")


def gradient_penalty(critic_model, interpolated_im, labels):
    # Returns the gradient penalty term in the loss for WGAN-GP
    critic = critic_model(interpolated_im, labels)
    grad = autograd.grad(
        outputs=critic,
        inputs=interpolated_im,
        grad_outputs=torch.ones_like(critic),
        create_graph=True,
        retain_graph=True,
    )[0]  # Returns a tuple of tensors with gradients
    # grad_outputs argument is used for performance improvement, unsure of the details.
    grad = grad.view(grad.shape[0], -1)
    return (grad.norm(2, dim=1) - 1) ** 2


def contractive_penalty(autoencoder, input_tensor):
    # Returns the contractive penalty term in the contractive autoencoder loss. Flattens the first dim of input_tensor
    input_tensor = input_tensor.flatten(start_dim=1)
    compute_jacobian = vmap(jacrev(autoencoder.encode, argnums=0), in_dims=0)
    jac = compute_jacobian(input_tensor)

    if len(jac.shape) == 2:
        # if the hidden dimension is one, an extra dimension has to be added to the Jacobian matrix
        batch_size, N = jac.shape
        jac = jac.view(batch_size, 1, N)
    return matrix_norm(jac, ord="fro", dim=(1, 2))


def contractive_penalty_v2(autoencoder, input_tensor, device):
    code = autoencoder.encode(input_tensor)
    jac = torch.zeros(code.shape[0], code.shape[1], input_tensor.shape[1]).to(device)

    for i in range(code.shape[1]):
        grad = autograd.grad(
            outputs=code[:, i],
            inputs=input_tensor,
            grad_outputs=torch.ones_like(code[:, i]),
            create_graph=True,
            retain_graph=True,
        )[0]
        jac[:, i, :] = grad

    if len(jac.shape) == 2:
        # if the hidden dimension is one, an extra dimension has to be added to the Jacobian matrix
        batch_size, N = jac.shape
        jac = jac.view(batch_size, 1, N)
    return matrix_norm(jac, ord="fro", dim=(1, 2))


def decoder_penalty(autoencoder, input_tensor):
    # Penalty similar to the contractive penalty but applied to the decoder. Applying this penalty encourages small
    # changes in code to corresponds to small changes in decoded vector.
    input_tensor = autoencoder.encode(input_tensor).flatten(start_dim=1)
    compute_jacobian = vmap(jacfwd(autoencoder.decode, argnums=0), in_dims=0)
    jac = compute_jacobian(input_tensor)

    if len(jac.shape) == 2:
        # if the hidden dimension is one, an extra dimension has to be added to the Jacobian matrix
        batch_size, N = jac.shape
        jac = jac.view(batch_size, 1, N)
    return matrix_norm(jac, ord="fro", dim=(1, 2))


def decoder_penalty_v2(autoencoder, input_tensor, device):
    code = autoencoder.encode(input_tensor)
    output_tensor = autoencoder.decode(code)
    jac = torch.zeros(output_tensor.shape[0], output_tensor.shape[1], code.shape[1]).to(
        device
    )

    for i in range(output_tensor.shape[1]):
        grad = autograd.grad(
            outputs=output_tensor[:, i],
            inputs=code,
            grad_outputs=torch.ones_like(output_tensor[:, i]),
            create_graph=True,
            retain_graph=True,
        )[0]
        jac[:, i, :] = grad

    if len(jac.shape) == 2:
        # if the hidden dimension is one, an extra dimension has to be added to the Jacobian matrix
        batch_size, N = jac.shape
        jac = jac.view(batch_size, 1, N)
    return matrix_norm(jac, ord="fro", dim=(1, 2))


def estimate_reconstruction_error(
    generator, data_loader, device, metric=nn.MSELoss(), n=30
):
    """
    This function computes the reconstruction MAE between real and fake images.
    This function also computes the reconstruction MAE using pixel with a
    structure / prediction of a structure.
    """
    generator.eval()
    z_dim = generator.get_z_dim()
    with torch.no_grad():
        # Get n samples of mean squared error between real and fake images.
        losses = torch.zeros(n, 1)
        struct_losses = torch.zeros(n, 1)
        for t in range(n):
            running_loss = 0.0
            struct_running_loss = 0.0
            it = 0
            for i, (real, labels) in enumerate(data_loader):
                real, labels = real.to(device), labels.to(device)
                batch_size = real.shape[0]
                z = torch.normal(0, 1, size=(batch_size, z_dim)).to(device)
                fake = generator(z, labels)

                # Part 1, compute full image reconstruction error
                running_loss += metric(real, fake)  # Mean over batch

                # Part 2, compute structural/masked reconstruction error
                # Mask out pixel with no structure/ no prediction of a structure
                real_binary_channel = real[:, 0, :, :].unsqueeze(1)
                fake_binary_channel = torch.round(fake[:, 0, :, :]).unsqueeze(1)
                mask = (real_binary_channel > -1.0) | (fake_binary_channel > -1.0)
                mask = mask.float()  # So that we can make the below calculations

                # Calculate a masked MAE
                abs_error = torch.abs(real - fake)
                if mask.sum() > 0:
                    number_of_channels = real.shape[1]
                    struct_mae = (abs_error * mask).sum() / (
                        mask.sum() * number_of_channels
                    )
                else:
                    struct_mae = 0
                struct_running_loss += struct_mae

                it += 1
            losses[t] = running_loss / it  # Average over batches
            struct_losses[t] = struct_running_loss / it
        # Return mean and variance of n samples...
        return (
            losses.mean().item(),
            losses.var().item(),
            struct_losses.mean().item(),
            struct_losses.var().item(),
        )


def estimate_forward_error(generator, data_loader, device, forward_network, n=30):
    # Estimates the forward of generator output with the help of a pretrained network
    generator.eval()
    forward_network.eval()
    z_dim = generator.get_z_dim()
    with torch.no_grad():
        # Get n samples of mean absolute error between real labels and fake labels
        # obtained from pretrained forward network.
        forward_error = torch.zeros(
            n, 1
        )  # Mean absolute difference between FEM-labels and fake forward labels.
        cnn_error = torch.zeros(
            n, 1
        )  # Mean absolute difference between real and fake forward labels.
        for t in range(n):
            running_loss = 0.0
            cnn_running_loss = 0.0
            it = 0
            for i, (real, labels) in enumerate(data_loader):
                real = real.to(device)
                labels = labels.to(device)
                batch_size = labels.shape[0]
                z = torch.normal(0, 1, size=(batch_size, z_dim)).to(device)
                fake = generator(z, labels)
                fake_forward_labels = forward_network(fake)
                real_forward_labels = forward_network(real)
                running_loss += (
                    (labels - fake_forward_labels).abs().mean()
                )  # Mean absolute error in batch
                cnn_running_loss += (
                    (real_forward_labels - fake_forward_labels).abs().mean()
                )
                it += 1
            forward_error[t] = running_loss / it  # Average over batch
            cnn_error[t] = cnn_running_loss / it
        # Return mean and variance of n samples...
        return forward_error.mean().item(), cnn_error.mean().item()


def construct_eff_net_sequence(
    module,
    in_channels,
    out_channels,
    stride,
    expansion,
    layers,
    activation,
    normalization_layer,
    is_transposed=False,
):
    if is_transposed:
        sequential = nn.Sequential(
            module(
                out_channels,
                in_channels,
                expansion_ratio=expansion,
                stride=1,
                kernel_size=3,
                padding=1,
                activation=activation,
                normalization_layer=normalization_layer,
            )
        )
        for layer in range(layers - 2):
            sequential.append(
                module(
                    in_channels,
                    in_channels,
                    expansion_ratio=expansion,
                    stride=1,
                    kernel_size=3,
                    padding=1,
                    activation=activation,
                    normalization_layer=normalization_layer,
                )
            )
        sequential.append(
            module(
                in_channels,
                in_channels,
                expansion_ratio=expansion,
                stride=stride,
                kernel_size=3,
                padding=1,
                activation=activation,
                normalization_layer=normalization_layer,
            )
        )
        return sequential
    else:
        # Construct one sequential block in efficient net and efficient net v2
        sequential = nn.Sequential(
            module(
                in_channels,
                in_channels,
                expansion_ratio=expansion,
                stride=stride,
                kernel_size=3,
                padding=1,
                activation=activation,
                normalization_layer=normalization_layer,
            )
        )
        for layer in range(layers - 2):
            sequential.append(
                module(
                    in_channels,
                    in_channels,
                    expansion_ratio=expansion,
                    stride=1,
                    kernel_size=3,
                    padding=1,
                    activation=activation,
                    normalization_layer=normalization_layer,
                )
            )
        sequential.append(
            module(
                in_channels,
                out_channels,
                expansion_ratio=expansion,
                stride=1,
                kernel_size=3,
                padding=1,
                activation=activation,
                normalization_layer=normalization_layer,
            )
        )
        return sequential


def print_losses(epoch, training_loss, validation_loss):
    print(
        "Epoch %d: Training loss = %.3E, Validation loss = %.3E"
        % (epoch + 1, training_loss, validation_loss)
    )


def print_losses_and_time(epoch, training_loss, validation_loss, time):
    print(
        "Epoch %d: Training loss = %.3E, Validation loss = %.3E, Epoch time: %.3f"
        % (epoch + 1, training_loss, validation_loss, time)
    )


def print_metric(epoch, training_metric, validation_metric, time):
    print(
        "Epoch %d: Metric training value = %.3E, Metric validation value = %.3E, Epoch time: %.3f"
        % (epoch + 1, training_metric, validation_metric, time)
    )


def print_gan_losses(epoch, losses, time):
    critic_loss, gen_loss = losses
    print(
        "Epoch %d: Critic loss = %.3E, Generator loss = %.3E, Epoch time: %.3f"
        % (epoch, critic_loss, gen_loss, time)
    )


def load_pretrained_h2y(filename, model):
    checkpoint = torch.load(filename, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def get_image_size(data_loader):
    im, _ = next(iter(data_loader))
    c = im.size(1)  # Channels
    h = im.size(2)  # Height
    w = im.size(3)  # Width

    return c, h, w


def get_label_size(data_loader):
    _, label = next(iter(data_loader))

    n = label.size(-1)
    if len(label.shape) == 2:
        c = 1
    else:
        c = label.size(-2)
    return c, n  # channels, number of elements


def moving_average(data, window_size):
    """
    Computes the moving average of a one dimensional array.
    Handles start/end by taking the average of fewer elements than window size.
    """
    window = np.ones(window_size)
    sums = np.convolve(data, window, mode="same")
    counts = np.convolve(np.ones_like(data), window, mode="same")
    return sums / counts


def gaussian_fun(x, magnitude, mu, sigma):
    g = norm.pdf(x, loc=mu, scale=sigma)
    return g * (magnitude / norm.pdf(mu, mu, sigma))


def find_generalization_gap_idx(train_err, val_err, threshold=0.1):
    """
    INPUTS:
        train_err: 1d array of size n
        val_err: 1d array of size n
    """
    assert train_err.ndim == 1 and val_err.ndim == 1
    diff = val_err - train_err
    percent_diff = diff / val_err
    return np.argwhere(percent_diff > threshold).flatten()[0]
