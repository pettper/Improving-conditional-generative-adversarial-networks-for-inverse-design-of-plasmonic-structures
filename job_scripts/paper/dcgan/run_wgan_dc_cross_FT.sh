#!/bin/bash
# Here you should put your own project id
#SBATCH -A hpc2n2024-042
# Allocate GPU and 14 CPU cores
#SBATCH --gres=gpu:v100:1
#SBATCH -c 14
# Ask for a suitable amount of time. Remember, this is the time the Jupyter notebook will be available!
#SBATCH --time=90:00:00
# Name of job
#SBATCH -J ml_pytorch
# Send mail when job begins
#SBATCH --mail-type=BEGIN
 
# Clear the environment from any previously loaded modules
module purge > /dev/null 2>&1
 
# Load the module environment suitable for the job
ml restore petter_modules

# Run WGANGP with FCGAN with architecture
arch=dc			# Model architecture
epochs=30000 		# Number of epochs to train
var=cross		# Input variables, absorption and scattering cross section
bs=64			# Batch size
lr=0.0001		# Learning rate for adam
lp=False		# Use label projection?
embed=True		# Use embedding network?
fn=saved_models/paper_results/gan_results/last_epoch_effv2_cross_lr00001_drop05.pth.tar		# Feed forward network to use as metric
save_name=saved_models/wgan_dc_cross_cylinder_FT.pth.tar
# load_name=saved_models/paper_results/gan_results/wgan_dc_cross_cylinder_FT.pth.tar
data_dir=data/dimer_cylinder_train_val_test/
python wgan_gp_main.py -a ${arch} -e ${epochs} -var ${var} -b ${bs} -lr ${lr} -lp ${lp} -em ${embed} -ffn ${fn} -sf ${save_name} -ddir ${data_dir}
