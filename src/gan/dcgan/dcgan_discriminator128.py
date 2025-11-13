from torch import nn
from torch.nn.utils.parametrizations import spectral_norm as sn

# Parameters
SLOPE = 0.2


class DCGANDiscriminator128(nn.Module):

    def __init__(self, inp_channels, y_dim, features=4, spectral_normalization=False):
        super().__init__()

        # Spectral normalization can be applied if specified
        if not spectral_normalization:
            def snorm(x):
                return x
        else:
            snorm = sn

        self.down_sampling_block = nn.Sequential(
            snorm(nn.Conv2d(inp_channels, features * 16, kernel_size=4, stride=2, padding=1, bias=False)),     # 64x64
            nn.LeakyReLU(SLOPE),
            snorm(nn.Conv2d(features * 16, features * 32, kernel_size=4, stride=2, padding=1, bias=False)),    # 32x32
            nn.InstanceNorm2d(features * 32, affine=True),
            nn.LeakyReLU(SLOPE),
            snorm(nn.Conv2d(features * 32, features * 64, kernel_size=4, stride=2, padding=1, bias=False)),    # 16x16
            nn.InstanceNorm2d(features * 64, affine=True),
            nn.LeakyReLU(SLOPE),
            snorm(nn.Conv2d(features * 64, features * 128, kernel_size=4, stride=2, padding=1, bias=False)),   # 8x8
            nn.InstanceNorm2d(features * 128, affine=True),
            nn.LeakyReLU(SLOPE),
            snorm(nn.Conv2d(features * 128, features * 256, kernel_size=4, stride=2, padding=1, bias=False)),  # 4x4
            nn.InstanceNorm2d(features * 256, affine=True),
            nn.LeakyReLU(SLOPE),
            snorm(nn.Conv2d(features * 256, y_dim, kernel_size=4, stride=2, padding=0, bias=False)),   # 1x1
            nn.Flatten(start_dim=2)
        )

    def forward(self, x):
        return self.down_sampling_block(x)
