import torch
import torch.nn as nn
'''
CBN (Conditional Batch Normalization layer)
    uses an MLP to predict the beta and gamma parameters in the batch norm equation
    Reference : https://papers.nips.cc/paper/7237-modulating-early-visual-processing-by-language.pdf
'''


class ConditionalBatchNorm(nn.Module):

    def __init__(self, num_features, embedding_size=20, hidden_size=200, eps=1e-5, momentum=0.9):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # Affine transform parameters
        self.gamma = nn.Parameter(torch.Tensor(num_features), requires_grad=True)
        self.beta = nn.Parameter(torch.Tensor(num_features), requires_grad=True)

        # Running mean and variance, these parameters are not trained by backprop
        self.running_mean = nn.Parameter(torch.Tensor(num_features), requires_grad=False)
        self.running_var = nn.Parameter(torch.Tensor(num_features), requires_grad=False)
        self.num_batches_tracked = nn.Parameter(torch.Tensor(1), requires_grad=False)

        # Parameter initilization
        self.reset_parameters()

        # MLP parameters
        self.emb_size = embedding_size
        self.hidden_size = hidden_size

        # MLP used to predict betas and gammas
        self.fc_gamma = nn.Sequential(
            nn.Linear(self.emb_size, self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, self.num_features),
        )

        self.fc_beta = nn.Sequential(
            nn.Linear(self.emb_size, self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, self.num_features),
        )

        # Initialize weights using Xavier initialization and biases with constant value
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.1)

    def reset_running_stats(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)
        self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        self.gamma.data.uniform_()
        self.beta.data.zero_()

    def forward(self, x, embedding):

        N, C, H, W = x.size()

        exponential_average_factor = 0.0
        if self.training:
            self.num_batches_tracked += 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked
            else:  # use exponential moving average
                exponential_average_factor = 1 - self.momentum

        # Obtain delta values from MLP
        delta_gamma = self.fc_gamma(embedding)
        delta_beta = self.fc_beta(embedding)

        gamma_cloned = self.gamma.clone()
        beta_cloned = self.beta.clone()

        gamma_cloned = gamma_cloned.view(1, C).expand(N, C)
        beta_cloned = beta_cloned.view(1, C).expand(N, C)

        # Update the values
        gamma_cloned = gamma_cloned + delta_gamma
        beta_cloned = beta_cloned + delta_beta

        # Standard batch normalization
        out, running_mean, running_var = batch_norm(x, self.running_mean, self.running_var, gamma_cloned,
                                                    beta_cloned,
                                                    self.training, exponential_average_factor, self.eps)

        if self.training:
            self.running_mean.data = running_mean.data
            self.running_var.data = running_var.data

        return out

def batch_norm(input, running_mean, running_var, gammas, betas,
               is_training, exponential_average_factor, eps):
    # Extract the dimensions
    N, C, H, W = input.size()

    # Mini-batch mean and variance
    input_channel_major = input.permute(1, 0, 2, 3).contiguous().view(input.size(1), -1)
    mean = input_channel_major.mean(dim=1)
    variance = input_channel_major.var(dim=1)

    # Normalize
    if is_training:

        # Compute running mean and variance
        running_mean = running_mean * (1 - exponential_average_factor) + mean * exponential_average_factor
        running_var = running_var * (1 - exponential_average_factor) + variance * exponential_average_factor

        # Training mode, normalize the data using its mean and variance
        X_hat = (input - mean.view(1, C, 1, 1).expand((N, C, H, W))) * 1.0 / torch.sqrt(
            variance.view(1, C, 1, 1).expand((N, C, H, W)) + eps)
    else:
        # Test mode, normalize the data using the running mean and variance
        X_hat = (input - running_mean.view(1, C, 1, 1).expand((N, C, H, W))) * 1.0 / torch.sqrt(
            running_var.view(1, C, 1, 1).expand((N, C, H, W)) + eps)

    # Scale and shift
    out = gammas.contiguous().view(N, C, 1, 1).expand((N, C, H, W)) * X_hat + betas.contiguous().view(N, C, 1,
                                                                                                      1).expand(
        (N, C, H, W))

    return out, running_mean, running_var