import torch
import numpy as np
from src.utils import gradient_penalty
from torch.linalg import vector_norm


def compute_soft_weight(label, noise_label, nu):
    # Computes the soft vicinal weight between label and noise label
    return torch.exp(-nu * vector_norm((noise_label - label) ** 2, dim=1, ord=2)).unsqueeze(1)    # Computes 2 norm


def get_fake_labels(noise_labels, device, nu, threshold):
    # Function that generates fake labels from noise labels according to the CcGAN-training algorithm
    # 'Threshold' and 'nu' are constant parameters
    offset = np.sqrt(-np.log(threshold) / nu)
    lower = noise_labels - offset
    upper = noise_labels + offset

    fake_labels = (upper - lower) * torch.rand(noise_labels.shape).to(device) + lower
    return fake_labels.clip(0.0, 1.0)


def get_fake_images(generator, labels, z_dim, device):
    # Function that generates fake images from provided fake labels
    # z_dim is the noise dimension passed to the generator
    batch_size = labels.shape[0]
    z = torch.normal(0.0, 1.0, size=(batch_size, z_dim)).to(device)
    fake_images = generator(z, labels)
    return fake_images


def get_noise_labels(labels, device, sigma):
    # Add noise to labels, clamp in interval [0, 1]
    eps_d = torch.normal(0.0, sigma, size=labels.shape).to(device)
    return (labels + eps_d).clamp(0.0, 1.0)


def compute_soft_vicinity_data(labels, noise_labels, autoencoder, device, batch_size, threshold, nu, sigma):
    # Function that extracts real images with noise labels in the soft vicinity of its real label.
    # Returns
    # 1. A list 'real_idx', with real image indices.
    # 2. A list 'noise_label_idx' with index corresponding to noise labels in the soft vicinity. Same length as real_idx
    # 3. A list with the soft weights corresponding to 'noise_label_idx'. Same length as real_idx
    real_idx = []
    real_soft_weights = []

    for i in range(batch_size):
        # Extract samples that are in the soft vicinity of the label
        in_vicinity = torch.where(vector_norm((labels - noise_labels[i]) ** 2, dim=1, ord=2) <=
                                  -torch.log(torch.Tensor([threshold])).item() / nu)[0]

        # While no label in the close vicinity is found, try again 10 times, otherwise accept original label
        it = 0
        while in_vicinity.shape[0] == 0 and it < 10:
            s = sigma / (2 ** it)   # Half the standard deviation of the noise each iteration and draw new noise label
            encoded = autoencoder.encode(labels[[i]])
            noise_labels[i] = autoencoder.decode((encoded + torch.normal(0.0, s, size=encoded.shape).to(device)).
                                                 clamp(0.0, 1.0))
            in_vicinity = torch.where(vector_norm((labels - noise_labels[i]) ** 2, dim=1, ord=2) <=
                                      -torch.log(torch.Tensor([threshold])).item() / nu)[0]
            it += 1

        if in_vicinity.shape[0] == 0:
            # No noise label found in vicinity, choose original label instead
            noise_labels[i] = labels[i]
            idx = i
            real_idx.append(idx)
        else:
            # Draw random image with noise label in close vicinity to the original label
            rand_idx = torch.randint(0, in_vicinity.shape[0], (1,)).item()
            idx = in_vicinity[rand_idx]
            real_idx.append(idx)

        # Compute the soft weight
        real_soft_weights.append(compute_soft_weight(noise_labels[[i]], labels[[idx]], nu))

    real_labels = noise_labels
    return real_idx, real_labels, torch.Tensor(real_soft_weights).unsqueeze(1).to(device)


def critic_loss(critic, device, real_images, fake_images, labels, real_soft_weights, fake_soft_weights,
                penalty_coefficient):
    # Function that computes the soft vicinal WGAN critic loss

    # Gradient penalty term
    b, c, h, w = real_images.shape
    eps = torch.rand(b, 1, 1, 1).repeat(1, c, h, w).to(device)
    interpolated = eps * real_images + (1 - eps) * fake_images
    gp = gradient_penalty(critic, interpolated, labels)

    # Compute critic loss, weighted Wasserstein distance with gradient penalty.
    real_critic_loss = real_soft_weights * critic(real_images, labels)
    fake_critic_loss = fake_soft_weights * critic(fake_images, labels)
    wasserstein_distance = real_critic_loss - fake_critic_loss
    return (-wasserstein_distance + penalty_coefficient * gp).mean(), wasserstein_distance.mean().item()

