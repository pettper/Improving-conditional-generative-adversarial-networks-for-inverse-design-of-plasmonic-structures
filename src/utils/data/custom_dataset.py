import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os

# Set numpy seed
np.random.seed(23)


class CustomDataset(Dataset):

    def __init__(self, root_dir, image_size, transform=None, target_transform=None):
        # root_dir is the directory where the data files are stored. Each file in the root directory corresponds to
        # exactly ONE sample image_res is a tuple of ints defining the image size of the provided image data e.g
        # (1, 64, 64) is a 64x64 image with one channel.
        self.root_dir = root_dir
        self.image_size = image_size
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(os.listdir(self.root_dir))

    def __getitem__(self, idx):
        # Get file with sample
        sample_file_str = os.listdir(self.root_dir)[idx]
        sample_path = os.path.join(self.root_dir, sample_file_str)
        # Get data from file
        image_data = pd.read_excel(sample_path, sheet_name='image_data', header=None)
        output_data = pd.read_excel(sample_path, sheet_name='output_data', header=None)

        # Construct sample
        image_data = torch.Tensor(image_data.iloc[:, 2].values).reshape(self.image_size)
        output_data = torch.Tensor(output_data.iloc[:, 1].values)

        if self.transform:
            image_data = self.transform(image_data)

        if self.target_transform:
            output_data = self.target_transform(output_data)

        return image_data, output_data

    def validation_split(self, val_ratio=0.2):
        n = self.__len__()  # sample size
        indices = np.arange(n)
        split_idx = np.int64(np.ceil(n * 0.2))

        if split_idx == n:
            raise Exception("Cannot split with {val_ratio}.")

        # Shuffle and split
        np.random.shuffle(indices)
        val_indices, train_indices = indices[:split_idx], indices[split_idx:]
        return train_indices, val_indices
