from src.utils import CustomDataset, CustomDatasetV2, DimerDataset, DimerVariable
from torch.utils.data import DataLoader
import time

N = 5
def data_loading_test(data_loader):
    # Test the average time it takes to load the dataset 5 times
    start = time.perf_counter()

    for _ in range(N):
        for i, sample in enumerate(data_loader):
            im, labels = sample

    end = time.perf_counter()
    elapsed_time = end - start
    return elapsed_time / N


xlsx_root_dir = "data/au_dimer_cylinder_data/xlsxfiles/"
feather_root_dir = "data/au_dimer_cylinder_data/featherfiles/"
DIM = 128
image_size = (1, DIM, DIM)

xlsx_dataset = CustomDataset(xlsx_root_dir, image_size, transform=None)
feather_dataset = CustomDatasetV2(feather_root_dir, image_size, transform=None)
tensor_dataset = DimerDataset(feather_root_dir, DimerVariable.ALL, transform=None)

xlsx_loader = DataLoader(xlsx_dataset, batch_size=DIM)
feather_loader = DataLoader(feather_dataset, batch_size=DIM)
tensor_loader = DataLoader(tensor_dataset, batch_size=DIM)

# Test
xlsx_time = data_loading_test(xlsx_loader)
feather_time = data_loading_test(feather_loader)
tensor_time = data_loading_test(tensor_loader)

print("Test result")
print(f".xlsx-file: {xlsx_time} s")
print(f".feather_files: {feather_time} s")
print(f".tensors: {tensor_time} s")
