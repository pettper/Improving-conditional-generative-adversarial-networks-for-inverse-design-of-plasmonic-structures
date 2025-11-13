# Help function library
import torch
from torch import autograd, nn
from torch.func import vmap, jacrev, jacfwd
from torch.linalg import matrix_norm


def count_parameters(model):
    # Returns the number of parameters in a pytorch model.
    params = sum(p.numel() for p in model.parameters())
    return params


def conv_output_size(input_size, kernel, padding, stride):
    # Assumes square input and kernel size
    return (input_size - kernel + 2 * padding) // stride + 1


def initialize_dcgan_weights(model):
    # Initializes weights according to the dcgan paper
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d, nn.InstanceNorm2d, nn.Linear)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


def gradient_penalty(critic_model, interpolated_im, labels):
    # Returns the gradient penalty term in the loss for WGAN-GP
    critic = critic_model(interpolated_im, labels)
    grad = autograd.grad(outputs=critic, inputs=interpolated_im, grad_outputs=torch.ones_like(critic),
                         create_graph=True, retain_graph=True)[0]  # Returns a tuple of tensors with gradients
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
    return matrix_norm(jac, ord='fro', dim=(1, 2))


def contractive_penalty_v2(autoencoder, input_tensor, device):

    code = autoencoder.encode(input_tensor)
    jac = torch.zeros(code.shape[0], code.shape[1], input_tensor.shape[1]).to(device)

    for i in range(code.shape[1]):
        grad = autograd.grad(outputs=code[:, i], inputs=input_tensor, grad_outputs=torch.ones_like(code[:, i]),
                             create_graph=True, retain_graph=True)[0]
        jac[:, i, :] = grad

    if len(jac.shape) == 2:
        # if the hidden dimension is one, an extra dimension has to be added to the Jacobian matrix
        batch_size, N = jac.shape
        jac = jac.view(batch_size, 1, N)
    return matrix_norm(jac, ord='fro', dim=(1, 2))


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
    return matrix_norm(jac, ord='fro', dim=(1, 2))


def decoder_penalty_v2(autoencoder, input_tensor, device):
    code = autoencoder.encode(input_tensor)
    output_tensor = autoencoder.decode(code)
    jac = torch.zeros(output_tensor.shape[0], output_tensor.shape[1], code.shape[1]).to(device)

    for i in range(output_tensor.shape[1]):
        grad = autograd.grad(outputs=output_tensor[:, i], inputs=code,
                             grad_outputs=torch.ones_like(output_tensor[:, i]), create_graph=True, retain_graph=True)[0]
        jac[:, i, :] = grad

    if len(jac.shape) == 2:
        # if the hidden dimension is one, an extra dimension has to be added to the Jacobian matrix
        batch_size, N = jac.shape
        jac = jac.view(batch_size, 1, N)
    return matrix_norm(jac, ord='fro', dim=(1, 2))


def estimate_reconstruction_error(generator, data_loader, device, metric=nn.MSELoss(), n=30):
    generator.eval()
    z_dim = generator.get_z_dim()
    with torch.no_grad():
        # Get n samples of mean squared error between real and fake images.
        losses = torch.zeros(n, 1)
        for t in range(n):
            running_loss = 0.0
            it = 0
            for i, (real, labels) in enumerate(data_loader):
                real, labels = real.to(device), labels.to(device)
                batch_size = real.shape[0]
                z = torch.normal(0, 1, size=(batch_size, z_dim)).to(device)
                fake = generator(z, labels)
                running_loss += metric(real, fake)  # Mean over batch
                it += 1
            losses[t] = running_loss / it   # Average over batches
        # Return mean and variance of n samples...
        return losses.mean().item(), losses.var().item()


def estimate_forward_error(generator, data_loader, device, forward_network, n=30):
    # Estimates the forward of generator output with the help of a pretrained network
    generator.eval()
    forward_network.eval()
    z_dim = generator.get_z_dim()
    with torch.no_grad():
        # Get n samples of mean absolute error between real labels and fake labels
        # obtained from pretrained forward network.
        forward_error = torch.zeros(n, 1)  # Mean absolute difference between FEM-labels and fake forward labels.
        cnn_error = torch.zeros(n, 1)  # Mean absolute difference between real and fake forward labels.
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
                running_loss += (labels - fake_forward_labels).abs().mean()     # Mean absolute error in batch
                cnn_running_loss += (real_forward_labels - fake_forward_labels).abs().mean()
                it += 1
            forward_error[t] = running_loss / it    # Average over batch
            cnn_error[t] = cnn_running_loss / it
        # Return mean and variance of n samples...
        return forward_error.mean().item(), cnn_error.mean().item()


def construct_eff_net_sequence(module, in_channels, out_channels, stride, expansion, layers, activation,
                               normalization_layer, is_transposed=False):
    if is_transposed:
        sequential = nn.Sequential(
            module(out_channels, in_channels, expansion_ratio=expansion, stride=1, kernel_size=3, padding=1,
                   activation=activation, normalization_layer=normalization_layer)
        )
        for layer in range(layers - 2):
            sequential.append(
                module(in_channels, in_channels, expansion_ratio=expansion, stride=1, kernel_size=3, padding=1,
                       activation=activation, normalization_layer=normalization_layer)
            )
        sequential.append(
            module(in_channels, in_channels, expansion_ratio=expansion, stride=stride, kernel_size=3, padding=1,
                   activation=activation, normalization_layer=normalization_layer)
        )
        return sequential
    else:
        # Construct one sequential block in efficient net and efficient net v2
        sequential = nn.Sequential(
            module(in_channels, in_channels, expansion_ratio=expansion, stride=stride, kernel_size=3, padding=1,
                   activation=activation, normalization_layer=normalization_layer)
        )
        for layer in range(layers - 2):
            sequential.append(
                module(in_channels, in_channels, expansion_ratio=expansion, stride=1, kernel_size=3, padding=1,
                       activation=activation, normalization_layer=normalization_layer)
            )
        sequential.append(
            module(in_channels, out_channels, expansion_ratio=expansion, stride=1, kernel_size=3, padding=1,
                   activation=activation, normalization_layer=normalization_layer)
        )
        return sequential


def print_losses(epoch, training_loss, validation_loss):
    print("Epoch %d: Training loss = %.3E, Validation loss = %.3E" % (
        epoch + 1, training_loss, validation_loss))


def print_losses_and_time(epoch, training_loss, validation_loss, time):
    print("Epoch %d: Training loss = %.3E, Validation loss = %.3E, Epoch time: %.3f" % (
        epoch + 1, training_loss, validation_loss, time))


def print_metric(epoch, training_metric, validation_metric, time):
    print("Epoch %d: Metric training value = %.3E, Metric validation value = %.3E, Epoch time: %.3f" % (
        epoch + 1, training_metric, validation_metric, time))


def print_gan_losses(epoch, losses, time):
    critic_loss, gen_loss = losses
    print("Epoch %d: Critic loss = %.3E, Generator loss = %.3E, Epoch time: %.3f" % (
        epoch, critic_loss, gen_loss, time))


def load_pretrained_h2y(filename, model):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model
