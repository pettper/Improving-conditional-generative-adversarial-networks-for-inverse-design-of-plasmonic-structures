#!/bin/bash
# Here you should put your own project id
#SBATCH -A hpc2n2026-031
# Ask for 1 Nvidia l40s gpu node
#SBATCH --gpus-per-node=l40s:1
#SBATCH -c 24
# Ask for a suitable amount of time. Remember, this is the time the Jupyter notebook will be available!
#SBATCH --time=00:05:00
# Name of job
#SBATCH -J ml_pytorch
# Send mail when job begins
#SBATCH --mail-type=BEGIN

# Clear the environment from any previously loaded modules
module purge > /dev/null 2>&1

# Load the module environment suitable for the job
ml GCCcore/12.3.0 GCC/12.3.0 OpenMPI/4.1.5
ml PyTorch-bundle/2.1.2-CUDA-12.1.1
ml Arrow/14.0.1 PyYAML/6.0 matplotlib/3.7.2 tensorboard/2.15.1

# Run WGANGP with FCGAN with architecture
./run_wgan_gp_runtime_test.sh
