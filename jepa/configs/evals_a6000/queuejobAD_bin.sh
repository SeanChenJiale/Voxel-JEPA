#!/bin/bash

# Initialize Conda (ensure this points to your Conda installation)
source ~/anaconda3/etc/profile.d/conda.sh

# Define the commands to run in each screen
commands=(
    # "conda activate jepa && cd /media/backup_16TB/sean/VoxelJEPA/jepa && python -m evals.main --fname /media/backup_16TB/sean/VoxelJEPA/jepa/configs/evals_a6000/vit_base/AD_bin/std_loss/12_1_std.yaml --devices cuda:2"
    # "conda activate jepa && cd /media/backup_16TB/sean/VoxelJEPA/jepa && python -m evals.main --fname /media/backup_16TB/sean/VoxelJEPA/jepa/configs/evals_a6000/vit_base/AD_bin_eval/std_loss/12_1_std.yaml --devices cuda:2 --validation True --plotter csv"
    # "conda activate jepa && cd /media/backup_16TB/sean/VoxelJEPA/jepa && python -m evals.main --fname /media/backup_16TB/sean/VoxelJEPA/jepa/configs/evals_a6000/vit_base/AD_bin_eval/std_loss/12_1_std_oasis.yaml --devices cuda:2 --validation True --plotter csv"
"conda activate jepa && cd /media/backup_16TB/sean/VoxelJEPA/jepa && python -m evals.main --fname /media/backup_16TB/sean/VoxelJEPA/jepa/configs/evals_a6000/vit_base/AD_bin/mean_loss/12_1_meanintensity_k400.yaml --devices cuda:0 --validation True --plotter csv"
"conda activate jepa && cd /media/backup_16TB/sean/VoxelJEPA/jepa && python -m evals.main --fname /media/backup_16TB/sean/VoxelJEPA/jepa/configs/evals_a6000/vit_base/AD_bin/mean_loss/12_1_meanintensity_no_ent.yaml --devices cuda:0 --validation True --plotter csv"
"conda activate jepa && cd /media/backup_16TB/sean/VoxelJEPA/jepa && python -m evals.main --fname /media/backup_16TB/sean/VoxelJEPA/jepa/configs/evals_a6000/vit_base/AD_bin/mean_loss/12_1_meanintensity.yaml --devices cuda:0 --validation True --plotter csv"
"conda activate jepa && cd /media/backup_16TB/sean/VoxelJEPA/jepa && python -m evals.main --fname /media/backup_16TB/sean/VoxelJEPA/jepa/configs/evals_a6000/vit_base/AD_bin/std_loss/12_1_std.yaml --devices cuda:0 --validation True --plotter csv"

)
# Loop through the commands and run them in sequence
for cmd in "${commands[@]}"; do
    echo "Running: $cmd"
    eval "$cmd"
done