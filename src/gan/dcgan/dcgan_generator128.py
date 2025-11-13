from torch import nn
from src.utils import ConditionalBatchNorm

# Parameters
HIDDEN_SIZE = 200


class DCGANGenerator128(nn.Module):
    # Class that implements a 128x128 DCGAN generator with or without conditional batch normalization
    def __init__(self, out_channels, features=4, use_cbn=False, embedding_size=20):
        super().__init__()
        self.use_cbn = use_cbn
        self.conv1 = nn.ConvTranspose2d(features * 256, features * 128, kernel_size=4, stride=2, padding=1,
                                        bias=False)
        self.conv2 = nn.ConvTranspose2d(features * 128, features * 64, kernel_size=4, stride=2, padding=1,
                                        bias=False)
        self.conv3 = nn.ConvTranspose2d(features * 64, features * 32, kernel_size=4, stride=2, padding=1,
                                        bias=False)
        self.conv4 = nn.ConvTranspose2d(features * 32, features * 16, kernel_size=4, stride=2, padding=1,
                                        bias=False)
        self.conv5 = nn.ConvTranspose2d(features * 16, out_channels, kernel_size=4, stride=2, padding=1,
                                        bias=False)
        self.activation = nn.ReLU()
        self.final_activation = nn.Tanh()
        if self.use_cbn:
            self.cbn1 = ConditionalBatchNorm(features * 128, embedding_size, HIDDEN_SIZE)
            self.cbn2 = ConditionalBatchNorm(features * 64, embedding_size, HIDDEN_SIZE)
            self.cbn3 = ConditionalBatchNorm(features * 32, embedding_size, HIDDEN_SIZE)
            self.cbn4 = ConditionalBatchNorm(features * 16, embedding_size, HIDDEN_SIZE)
        else:
            self.bn1 = nn.BatchNorm2d(features * 128)
            self.bn2 = nn.BatchNorm2d(features * 64)
            self.bn3 = nn.BatchNorm2d(features * 32)
            self.bn4 = nn.BatchNorm2d(features * 16)

    def forward(self, x, embedding=None):
        # An embedding has to be provided if use_cbn is set to true
        if self.use_cbn:
            x = self.activation(self.cbn1(self.conv1(x), embedding))
            x = self.activation(self.cbn2(self.conv2(x), embedding))
            x = self.activation(self.cbn3(self.conv3(x), embedding))
            x = self.activation(self.cbn4(self.conv4(x), embedding))
            return self.final_activation(self.conv5(x))
        else:
            x = self.activation(self.bn1(self.conv1(x)))
            x = self.activation(self.bn2(self.conv2(x)))
            x = self.activation(self.bn3(self.conv3(x)))
            x = self.activation(self.bn4(self.conv4(x)))
            return self.final_activation(self.conv5(x))
