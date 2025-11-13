import pandas as pd
import os

data_root = "../anisotropic_au_structures/test/"
xlsx_dir = data_root + "xlsxfiles/"
feather_dir = data_root + "featherfiles/"
files = os.listdir(xlsx_dir)

if not os.path.exists(feather_dir):
    os.mkdir(feather_dir)

for i, file in enumerate(files):
    sample_path = xlsx_dir + file
    sample_name = file[:-5]  # Remove .xlsx from name
    path = feather_dir + sample_name

    if not os.path.exists(path):
        os.mkdir(path)

    image_data = pd.read_excel(sample_path, sheet_name='image_data', header=None)
    output_data = pd.read_excel(sample_path, sheet_name='output_data', header=None)

    # Write dataframe to feather format
    image_data.to_feather(path + "/image.feather")
    output_data.to_feather(path + "/spectrum.feather")

    if i % 20 == 0:
        print(f"Done with {i+1} files.")
