#!/bin/bash

# Initialize Conda (ensure this points to your Conda installation)
source ~/anaconda3/etc/profile.d/conda.sh

# Define the commands to run in each screen
commands=(
    #diag MCICN
    # "conda activate jepa && cd /media/backup_16TB/sean/VoxelJEPA/jepa && python -m evals.main --fname /media/backup_16TB/sean/VoxelJEPA/jepa/configs/evals_a6000/vit_base/MCI_CN/combin_loss/12_1_0_15std_0_3_mean_diag.yaml --devices cuda:0"
    # "conda activate jepa && cd /media/backup_16TB/sean/VoxelJEPA/jepa && python -m evals.main --fname /media/backup_16TB/sean/VoxelJEPA/jepa/configs/evals_a6000/vit_base/MCI_CN/combin_loss/12_1_0_15std_0_3_mean_diag.yaml --devices cuda:0 --validation True --plotter csv"
    # "conda activate jepa && cd /media/backup_16TB/sean/VoxelJEPA/jepa && python -m evals.main --fname /media/backup_16TB/sean/VoxelJEPA/jepa/configs/evals_a6000/vit_base/MCI_CN/combin_loss/12_1_0_15std_0_3_mean_diag.yaml --devices cuda:0 --validation True --plotter csv"
    #diag ADMCICN
    # "conda activate jepa && cd /media/backup_16TB/sean/VoxelJEPA/jepa && python -m evals.main --fname /media/backup_16TB/sean/VoxelJEPA/jepa/configs/evals_a6000/vit_base/ADCNMCI/combin_loss/12_1_0_15std_0_3_mean_diag.yaml --devices cuda:0"
    # "conda activate jepa && cd /media/backup_16TB/sean/VoxelJEPA/jepa && python -m evals.main --fname /media/backup_16TB/sean/VoxelJEPA/jepa/configs/evals_a6000/vit_base/ADCNMCI/combin_loss/12_1_0_15std_0_3_mean_diag.yaml --devices cuda:0 --validation True --plotter csv"
    # "conda activate jepa && cd /media/backup_16TB/sean/VoxelJEPA/jepa && python -m evals.main --fname /media/backup_16TB/sean/VoxelJEPA/jepa/configs/evals_a6000/vit_base/ADCNMCI_eval/combin_loss/12_1_0_15std_0_3_mean_diag.yaml --devices cuda:0 --validation True --plotter csv"
    #trace ADMCICN
    "conda activate jepa && cd /media/backup_16TB/sean/VoxelJEPA/jepa && python -m evals.main --fname /media/backup_16TB/sean/VoxelJEPA/jepa/configs/evals_a6000/vit_base/ADCNMCI/combin_loss/12_1_0_05std_0_1_mean_trace.yaml --devices cuda:0"
    "conda activate jepa && cd /media/backup_16TB/sean/VoxelJEPA/jepa && python -m evals.main --fname /media/backup_16TB/sean/VoxelJEPA/jepa/configs/evals_a6000/vit_base/ADCNMCI/combin_loss/12_1_0_05std_0_1_mean_trace.yaml --devices cuda:0 --validation True --plotter csv"
    "conda activate jepa && cd /media/backup_16TB/sean/VoxelJEPA/jepa && python -m evals.main --fname /media/backup_16TB/sean/VoxelJEPA/jepa/configs/evals_a6000/vit_base/ADCNMCI_eval/combin_loss/12_1_0_05std_0_1_mean_trace.yaml --devices cuda:0 --validation True --plotter csv"



    # "conda activate jepa && cd /media/backup_16TB/sean/VoxelJEPA/jepa && python -m evals.main --fname /media/backup_16TB/sean/VoxelJEPA/jepa/configs/evals_a6000/vit_base/FT_Bin/combin_loss/12_1_0_05std_0_1_mean_trace.yaml --devices cuda:0"
    # "conda activate jepa && cd /media/backup_16TB/sean/VoxelJEPA/jepa && python -m evals.main --fname /media/backup_16TB/sean/VoxelJEPA/jepa/configs/evals_a6000/vit_base/FT_Bin/combin_loss/12_1_0_05std_0_1_mean_trace.yaml --devices cuda:0 --validation True --plotter csv"
    # "conda activate jepa && cd /media/backup_16TB/sean/VoxelJEPA/jepa && python -m evals.main --fname /media/backup_16TB/sean/VoxelJEPA/jepa/configs/evals_a6000/vit_base/FT_Bin_eval/combin_loss/12_1_0_05std_0_1_mean_trace.yaml --devices cuda:0 --validation True --plotter csv"

)

# Loop through the commands and run them in sequence
for cmd in "${commands[@]}"; do
    echo "Running: $cmd"
    eval "$cmd"
done