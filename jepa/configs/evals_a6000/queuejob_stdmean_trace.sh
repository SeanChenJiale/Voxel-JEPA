#!/bin/bash

# Initialize Conda (ensure this points to your Conda installation)
source ~/anaconda3/etc/profile.d/conda.sh

# Define the commands to run in each screen
commands=(
    #diag MCICN
    # "conda activate jepa && cd /media/backup_16TB/sean/VoxelJEPA/jepa && python -m evals.main --fname /media/backup_16TB/sean/VoxelJEPA/jepa/configs/evals_a6000/vit_base/MCI_CN/std_loss/12_1_std_0_075.yaml --devices cuda:0"
    # "conda activate jepa && cd /media/backup_16TB/sean/VoxelJEPA/jepa && python -m evals.main --fname /media/backup_16TB/sean/VoxelJEPA/jepa/configs/evals_a6000/vit_base/MCI_CN/std_loss/12_1_std_0_075.yaml --devices cuda:0 --validation True --plotter csv"
    # "conda activate jepa && cd /media/backup_16TB/sean/VoxelJEPA/jepa && python -m evals.main --fname /media/backup_16TB/sean/VoxelJEPA/jepa/configs/evals_a6000/vit_base/MCI_CN_eval/std_loss/12_1_std_0_075.yaml --devices cuda:0 --validation True --plotter csv"

    #diag MCICN
    # "conda activate jepa && cd /media/backup_16TB/sean/VoxelJEPA/jepa && python -m evals.main --fname /media/backup_16TB/sean/VoxelJEPA/jepa/configs/evals_a6000/vit_base/MCI_CN/combin_loss/12_1_0_3std_0_6_mean_diag.yaml --devices cuda:0"
    # "conda activate jepa && cd /media/backup_16TB/sean/VoxelJEPA/jepa && python -m evals.main --fname /media/backup_16TB/sean/VoxelJEPA/jepa/configs/evals_a6000/vit_base/MCI_CN/combin_loss/12_1_0_3std_0_6_mean_diag.yaml --devices cuda:0 --validation True --plotter csv"
    # "conda activate jepa && cd /media/backup_16TB/sean/VoxelJEPA/jepa && python -m evals.main --fname /media/backup_16TB/sean/VoxelJEPA/jepa/configs/evals_a6000/vit_base/MCI_CN_eval/combin_loss/12_1_0_3std_0_6_mean_diag.yaml --devices cuda:0 --validation True --plotter csv"


)

# Loop through the commands and run them in sequence
for cmd in "${commands[@]}"; do
    echo "Running: $cmd"
    eval "$cmd"
done