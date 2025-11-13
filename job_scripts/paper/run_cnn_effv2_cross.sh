#!/bin/bash
# Here you should put your own project id
#SBATCH -A hpc2n2024-042
# Allocate GPU and 14 CPU cores
#SBATCH --gres=gpu:v100:1
#SBATCH -c 14
# Ask for a suitable amount of time. Remember, this is the time the Jupyter notebook will be available!
#SBATCH --time=50:00:00
# Name of job
#SBATCH -J ml_pytorch
# Send mail when job begins
#SBATCH --mail-type=BEGIN
 
# Clear the environment from any previously loaded modules
module purge > /dev/null 2>&1
 
# Load the module environment suitable for the job
ml restore petter_modules

# Arguments
dropout=0.5 	# Dropout rate in model
epochs=50000	# Number of epochs to train
batch=64	# Batch size
arch=effv2	# Model architecture
var=cross	# Target variable
lr=0.0001	# Learning rate
save_filename=saved_models/effv2_cross_all_structures_lr00001_drop05.pth.tar	# Filename to save model to
data_root=data/anisotropic_au_structures_train_val_test/			# Data directory root

# Run CNN-model
python cnn_main.py -a ${arch} -var ${var} -e ${epochs} -b ${batch} -lr ${lr} -dr ${dropout} -sf ${save_filename} -ddir ${data_root}
