import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.utils.data.dimer_variables import DimerVariable

# Set numpy seed
np.random.seed(23)

# CONSTANTS
PIXEL_MIN, PIXEL_MAX = -1.0, 1.0
Y_DIM = 81  # Sample points in target data. (Spectral data with 81 discrete points)
C, W, H = 2, 128, 128  # Channels, Pixels, Pixels


class DimerDataset(Dataset):
    def __init__(
        self,
        root_dir,
        variable,
        transform=None,
        target_transform=None,
        inverse_transform=None,
        inverse_target_transform=None,
        number_of_samples=None,
    ):
        """
        root_dir: root_dir is the directory where the data files are stored. Each file in the root directory
        corresponds to exactly ONE sample.
        variable: enum specifying what variable to use
        y_dim: Number of points in output_data/variable
        transform: Optional transform to the image data
        target_transform: Optional transform to the output data
        inverse_transform: Optional inverse transform to the image data
        inverse_target_transform: Optional inverse transform to the output data
        number_of_samples: Optional to specify size of dataset. Then __len__ will return number_of_samples.
        """
        # Expects 2 channels with 128x128
        self.root_dir = root_dir
        self.variable = variable
        self.transform = transform
        self.inverse_transform = inverse_transform
        self.target_transform = target_transform
        self.inverse_target_transform = inverse_target_transform
        self.number_of_samples = number_of_samples

        # Data variables to be returned
        self.output_min, self.output_data = 0, 0
        self.shape_min, self.shape_max = 0, 0
        self.top_min, self.top_max = 0, 0
        self.sca_min, self.sca_max = 0, 0
        self.abs_min, self.abs_max = 0, 0
        self.rot_min, self.rot_max = 0, 0
        self.elip_min, self.elip_max = 0, 0

        sorted_root_dir = sorted(os.listdir(self.root_dir))
        N = self.__len__()
        self.image_data = torch.empty(N, C, W, H)

        # Create target data structure
        if (
            self.variable == DimerVariable.ALL
            or self.variable == DimerVariable.CROSS_SECTIONS
        ):
            self.output_data = torch.empty(N, self.variable.size(), Y_DIM)
        else:
            self.output_data = torch.empty(N, Y_DIM)

        # Load all samples
        for idx in range(N):
            sample_dir = sorted_root_dir[idx]
            sample_path = os.path.join(self.root_dir, sample_dir)
            sample_files = sorted(os.listdir(sample_path))

            image_file_path = os.path.join(sample_path, sample_files[0])
            spectrum_file_path = os.path.join(sample_path, sample_files[1])

            # Get data from files
            image_data = pd.read_feather(image_file_path)
            output_data = pd.read_feather(spectrum_file_path)

            if output_data.isnull().values.any():
                print("output data has Nan")
            if image_data.isnull().values.any():
                print("image data has Nan")

            # Construct sample
            # isAu channel
            self.image_data[idx, 0, :, :] = torch.Tensor(
                image_data.iloc[:, 2].values
            ).reshape((1, 128, 128))
            # topylogy channel
            self.image_data[idx, 1, :, :] = torch.Tensor(
                image_data.iloc[:, 3].values
            ).reshape((1, 128, 128))
            # output variable
            if (
                self.variable == DimerVariable.ALL
                or self.variable == DimerVariable.CROSS_SECTIONS
            ):
                self.output_data[idx, :] = torch.Tensor(
                    output_data.iloc[:, self.variable.value].values
                ).transpose(0, 1)
            else:
                self.output_data[idx, :] = torch.Tensor(
                    output_data.iloc[:, self.variable.value].values
                )

        # Store max and min data
        # Image
        self.shape_min = self.image_data[:, 0, :, :].min().item()
        self.shape_max = self.image_data[:, 0, :, :].max().item()
        self.top_min = self.image_data[:, 1, :, :].min().item()
        self.top_max = self.image_data[:, 1, :, :].max().item()
        # Targets
        if (
            self.variable == DimerVariable.CROSS_SECTIONS
            or self.variable == DimerVariable.ALL
        ):
            self.sca_min = self.output_data[:, 0, :].min().item()
            self.sca_max = self.output_data[:, 0, :].max().item()
            self.abs_min = self.output_data[:, 1, :].min().item()
            self.abs_max = self.output_data[:, 1, :].max().item()
        else:
            self.output_min = self.output_data.min().item()
            self.output_max = self.output_data.max().item()
        if self.variable == DimerVariable.ALL:
            self.rot_min = self.output_data[:, 2, :].min().item()
            self.rot_max = self.output_data[:, 2, :].max().item()
            self.elip_min = self.output_data[:, 3, :].min().item()
            self.elip_max = self.output_data[:, 3, :].max().item()

        # Transform
        self.image_data = self.apply_transform(self.image_data)
        self.output_data = self.apply_target_transform(self.output_data)

    def __len__(self):
        total_number_samples = len(os.listdir(self.root_dir))
        if self.number_of_samples is None:
            n = total_number_samples
        else:
            n = torch.min(torch.Tensor([total_number_samples, self.number_of_samples]))
        assert n > 0
        return int(n)

    def __getitem__(self, idx):
        # Get file with sample
        image_data = self.image_data[idx]
        output_data = self.output_data[idx]

        return image_data, output_data

    def apply_transform(self, image_data):
        # Image data transform
        if self.transform:
            image_data = self.transform(image_data)
        else:
            # Default image transform
            image_data = self.default_transform(image_data)
        return image_data

    def apply_inverse_transform(self, image_data):
        if self.inverse_transform:
            image_data = self.inverse_transform(image_data)
        else:
            image_data = self.default_inverse_transform(image_data)
        return image_data

    def apply_target_transform(self, output_data):
        # Target transform
        if self.target_transform:
            output_data = self.target_transform(output_data)
        else:
            # Default transform
            output_data = self.default_target_transform(output_data)
        return output_data

    def apply_inverse_target_transform(self, output_data):
        if self.inverse_target_transform:
            output_data = self.inverse_target_transform(output_data)
        else:
            output_data = self.default_inverse_target_transform(output_data)
        return output_data

    def default_transform(self, image_data):
        image_data[:, 0, :, :] = PIXEL_MIN + (
            image_data[:, 0, :, :] - self.shape_min
        ) * (PIXEL_MAX - PIXEL_MIN) / (self.shape_max - self.shape_min)
        image_data[:, 1, :, :] = PIXEL_MIN + (image_data[:, 1, :, :] - self.top_min) * (
            PIXEL_MAX - PIXEL_MIN
        ) / (self.top_max - self.top_min)
        return image_data

    def default_inverse_transform(self, image_data):
        image_data[:, 0, :, :] = (image_data[:, 0, :, :] - PIXEL_MIN) * (
            self.shape_max - self.shape_min
        ) / (PIXEL_MAX - PIXEL_MIN) + self.shape_min
        image_data[:, 1, :, :] = (image_data[:, 1, :, :] - PIXEL_MIN) * (
            self.top_max - self.top_min
        ) / (PIXEL_MAX - PIXEL_MIN) + self.top_min
        return image_data

    def default_target_transform(self, output_data):
        if (
            self.variable == DimerVariable.ALL
            or self.variable == DimerVariable.CROSS_SECTIONS
        ):
            output_data[:, 0, :] = (output_data[:, 0, :] - self.sca_min) / (
                self.sca_max - self.sca_min
            )
            output_data[:, 1, :] = (output_data[:, 1, :] - self.abs_min) / (
                self.abs_max - self.abs_min
            )
        if self.variable == DimerVariable.ALL:
            output_data[:, 2, :] = (output_data[:, 2, :] - self.rot_min) / (
                self.rot_max - self.rot_min
            )
            output_data[:, 3, :] = (output_data[:, 3, :] - self.elip_min) / (
                self.elip_max - self.elip_min
            )
        else:
            if self.variable == DimerVariable.SCATTERING_CROSS_SECTION:
                output_data = (output_data - self.sca_min) / (
                    self.sca_max - self.sca_min
                )
            elif self.variable == DimerVariable.ABSORPTION_CROSS_SECTION:
                output_data = (output_data - self.abs_min) / (
                    self.abs_max - self.abs_min
                )
            elif self.variable == DimerVariable.ROTATION:
                output_data = (output_data - self.rot_min) / (
                    self.rot_max - self.rot_min
                )
            elif self.variable == DimerVariable.ELLIPTICITY:
                output_data = (output_data - self.elip_min) / (
                    self.elip_max - self.elip_min
                )
        return output_data

    def default_inverse_target_transform(self, output_data):
        if self.variable == DimerVariable.ALL:
            output_data[:, 0, :] = (
                output_data[:, 0, :] * (self.sca_max - self.sca_min) + self.sca_min
            )
            output_data[:, 1, :] = (
                output_data[:, 1, :] * (self.abs_max - self.abs_min) + self.abs_min
            )
            output_data[:, 2, :] = (
                output_data[:, 2, :] * (self.rot_max - self.rot_min) + self.rot_min
            )
            output_data[:, 3, :] = (
                output_data[:, 3, :] * (self.elip_max - self.elip_min) + self.elip_min
            )
        elif self.variable == DimerVariable.CROSS_SECTIONS:
            output_data[:, 0, :] = (
                output_data[:, 0, :] * (self.sca_max - self.sca_min) + self.sca_min
            )
            output_data[:, 1, :] = (
                output_data[:, 1, :] * (self.abs_max - self.abs_min) + self.abs_min
            )
        else:
            if self.variable == DimerVariable.SCATTERING_CROSS_SECTION:
                output_data = output_data * (self.sca_max - self.sca_min) + self.sca_min
            elif self.variable == DimerVariable.ABSORPTION_CROSS_SECTION:
                output_data = output_data * (self.abs_max - self.abs_min) + self.abs_min
            elif self.variable == DimerVariable.ROTATION:
                output_data = output_data * (self.rot_max - self.rot_min) + self.rot_min
            elif self.variable == DimerVariable.ELLIPTICITY:
                output_data = (
                    output_data * (self.elip_max - self.elip_min) + self.elip_min
                )
        return output_data

    def validation_split(self, val_ratio=0.2):
        n = self.__len__()  # sample size
        indices = np.arange(n)
        split_idx = np.int64(np.ceil(n * val_ratio))

        if split_idx == n:
            raise Exception("Cannot split with {val_ratio}.")

        # Shuffle and split
        np.random.shuffle(indices)
        val_indices, train_indices = indices[:split_idx], indices[split_idx:]
        return train_indices, val_indices

    def get_min_and_max_data(self):
        # Returns the minimum and maximum values of image data and output data before any transforms take place
        if self.variable == DimerVariable.ALL:
            return (self.shape_min, self.shape_max, self.top_min, self.top_max), (
                self.sca_min,
                self.sca_max,
                self.abs_min,
                self.abs_max,
                self.rot_min,
                self.rot_max,
                self.elip_min,
                self.elip_max,
            )
        elif self.variable == DimerVariable.CROSS_SECTIONS:
            return (self.shape_min, self.shape_max, self.top_min, self.top_max), (
                self.sca_min,
                self.sca_max,
                self.abs_min,
                self.abs_max,
            )
        else:
            return (self.shape_min, self.shape_max, self.top_min, self.top_max), (
                self.output_min,
                self.output_max,
            )
