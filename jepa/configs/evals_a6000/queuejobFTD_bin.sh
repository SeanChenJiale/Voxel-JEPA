#!/bin/bash

# Initialize Conda (ensure this points to your Conda installation)
source ~/anaconda3/etc/profile.d/conda.sh

# Define the commands to run in each screen
commands=(
    "conda activate jepa && cd /media/backup_16TB/sean/VoxelJEPA/jepa && python -m evals.main --fname /media/backup_16TB/sean/VoxelJEPA/jepa/configs/evals_a6000/vit_base/FT_Bin/allscantrain_singletestval/12_1_meanintensity_k400.yaml --devices cuda:1"
    "conda activate jepa && cd /media/backup_16TB/sean/VoxelJEPA/jepa && python -m evals.main --fname /media/backup_16TB/sean/VoxelJEPA/jepa/configs/evals_a6000/vit_base/FT_Bin/allscantrain_singletestval/12_1_meanintensity_no_ent.yaml --devices cuda:1"
    "conda activate jepa && cd /media/backup_16TB/sean/VoxelJEPA/jepa && python -m evals.main --fname /media/backup_16TB/sean/VoxelJEPA/jepa/configs/evals_a6000/vit_base/FT_Bin/allscantrain_singletestval/12_1_meanintensity.yaml --devices cuda:1"
    "conda activate jepa && cd /media/backup_16TB/sean/VoxelJEPA/jepa && python -m evals.main --fname /media/backup_16TB/sean/VoxelJEPA/jepa/configs/evals_a6000/vit_base/FT_Bin/allscantrain_singletestval/12_1_std.yaml --devices cuda:1"

    "conda activate jepa && cd /media/backup_16TB/sean/VoxelJEPA/jepa && python -m evals.main --fname /media/backup_16TB/sean/VoxelJEPA/jepa/configs/evals_a6000/vit_base/FT_Bin_eval/allscantrain_singletestval/12_1_meanintensity_k400.yaml --devices cuda:1 --validation True --plotter csv"
    "conda activate jepa && cd /media/backup_16TB/sean/VoxelJEPA/jepa && python -m evals.main --fname /media/backup_16TB/sean/VoxelJEPA/jepa/configs/evals_a6000/vit_base/FT_Bin_eval/allscantrain_singletestval/12_1_meanintensity_no_ent.yaml --devices cuda:1 --validation True --plotter csv"
    "conda activate jepa && cd /media/backup_16TB/sean/VoxelJEPA/jepa && python -m evals.main --fname /media/backup_16TB/sean/VoxelJEPA/jepa/configs/evals_a6000/vit_base/FT_Bin_eval/allscantrain_singletestval/12_1_meanintensity.yaml --devices cuda:1 --validation True --plotter csv"
    "conda activate jepa && cd /media/backup_16TB/sean/VoxelJEPA/jepa && python -m evals.main --fname /media/backup_16TB/sean/VoxelJEPA/jepa/configs/evals_a6000/vit_base/FT_Bin_eval/allscantrain_singletestval/12_1_std.yaml --devices cuda:1 --validation True --plotter csv"
)

# Loop through the commands and run them in sequence
for cmd in "${commands[@]}"; do
    echo "Running: $cmd"
    eval "$cmd"
done