#!/bin/bash

# Initialize Conda (ensure this points to your Conda installation)
source ~/anaconda3/etc/profile.d/conda.sh

# "conda activate jepa && cd /media/backup_16TB/sean/VoxelJEPA/jepa && python -m evals.main --fname /media/backup_16TB/sean/VoxelJEPA/jepa/configs/evals_a6000/vit_base/MCI_CN/std_loss/12_1_std.yaml --devices cuda:0"
# "conda activate jepa && cd /media/backup_16TB/sean/VoxelJEPA/jepa && python -m evals.main --fname /media/backup_16TB/sean/VoxelJEPA/jepa/configs/evals_a6000/vit_base/ADCNMCI/std_loss/12_1_std.yaml --devices cuda:0"

# Define the commands to run in each screen
commands=(

# "conda activate jepa && cd /media/backup_16TB/sean/VoxelJEPA/jepa && python -m evals.main --fname /media/backup_16TB/sean/VoxelJEPA/jepa/configs/evals_a6000/vit_base/MCI_CN/std_loss/12_1_std.yaml --devices cuda:0 --validation True --plotter csv"
# "conda activate jepa && cd /media/backup_16TB/sean/VoxelJEPA/jepa && python -m evals.main --fname /media/backup_16TB/sean/VoxelJEPA/jepa/configs/evals_a6000/vit_base/ADCNMCI/std_loss/12_1_std.yaml --devices cuda:0 --validation True --plotter csv"
# "conda activate jepa && cd /media/backup_16TB/sean/VoxelJEPA/jepa && python -m evals.main --fname /media/backup_16TB/sean/VoxelJEPA/jepa/configs/evals_a6000/vit_base/MCI_CN_eval/std_loss/12_1_std.yaml --devices cuda:0 --validation True --plotter csv"
# "conda activate jepa && cd /media/backup_16TB/sean/VoxelJEPA/jepa && python -m evals.main --fname /media/backup_16TB/sean/VoxelJEPA/jepa/configs/evals_a6000/vit_base/ADCNMCI_eval/std_loss/12_1_std.yaml --devices cuda:0 --validation True --plotter csv"
# "conda activate jepa && cd /media/backup_16TB/sean/VoxelJEPA/jepa && python -m evals.main --fname /media/backup_16TB/sean/VoxelJEPA/jepa/configs/evals_a6000/vit_base/ADCNMCI/12_1_meanintensity_k400.yaml --devices cuda:0 --validation True --plotter csv"
# "conda activate jepa && cd /media/backup_16TB/sean/VoxelJEPA/jepa && python -m evals.main --fname /media/backup_16TB/sean/VoxelJEPA/jepa/configs/evals_a6000/vit_base/ADCNMCI/12_1_meanintensity.yaml --devices cuda:0 --validation True --plotter csv"
# "conda activate jepa && cd /media/backup_16TB/sean/VoxelJEPA/jepa && python -m evals.main --fname /media/backup_16TB/sean/VoxelJEPA/jepa/configs/evals_a6000/vit_base/ADCNMCI/12_1_no_meanintensity.yaml --devices cuda:0 --validation True --plotter csv"

# "conda activate jepa && cd /media/backup_16TB/sean/VoxelJEPA/jepa && python -m evals.main --fname /media/backup_16TB/sean/VoxelJEPA/jepa/configs/evals_a6000/vit_base/MCI_CN/mean_loss/12_1_meanintensity_k400.yaml --devices cuda:0 --validation True --plotter csv"
# "conda activate jepa && cd /media/backup_16TB/sean/VoxelJEPA/jepa && python -m evals.main --fname /media/backup_16TB/sean/VoxelJEPA/jepa/configs/evals_a6000/vit_base/MCI_CN/mean_loss/12_1_meanintensity.yaml --devices cuda:0 --validation True --plotter csv"
# "conda activate jepa && cd /media/backup_16TB/sean/VoxelJEPA/jepa && python -m evals.main --fname /media/backup_16TB/sean/VoxelJEPA/jepa/configs/evals_a6000/vit_base/MCI_CN/mean_loss/12_1_no_meanintensity.yaml --devices cuda:0 --validation True --plotter csv"

    # RAN
    # "conda activate jepa && cd /media/backup_16TB/sean/VoxelJEPA/jepa && python -m evals.main --fname /media/backup_16TB/sean/VoxelJEPA/jepa/configs/evals_a6000/vit_base/FT_Bin/allscantrain_singletestval/12_1_meanintensity_k400.yaml --devices cuda:1 --validation True --plotter csv"
    # "conda activate jepa && cd /media/backup_16TB/sean/VoxelJEPA/jepa && python -m evals.main --fname /media/backup_16TB/sean/VoxelJEPA/jepa/configs/evals_a6000/vit_base/FT_Bin/allscantrain_singletestval/12_1_meanintensity_no_ent.yaml --devices cuda:1 --validation True --plotter csv"
    # "conda activate jepa && cd /media/backup_16TB/sean/VoxelJEPA/jepa && python -m evals.main --fname /media/backup_16TB/sean/VoxelJEPA/jepa/configs/evals_a6000/vit_base/FT_Bin_eval/allscantrain_singletestval/12_1_meanintensity_k400.yaml --devices cuda:1 --validation True --plotter csv"
    # "conda activate jepa && cd /media/backup_16TB/sean/VoxelJEPA/jepa && python -m evals.main --fname /media/backup_16TB/sean/VoxelJEPA/jepa/configs/evals_a6000/vit_base/FT_Bin_eval/allscantrain_singletestval/12_1_meanintensity_no_ent.yaml --devices cuda:1 --validation True --plotter csv"

    # "conda activate jepa && cd /media/backup_16TB/sean/VoxelJEPA/jepa && python -m evals.main --fname /media/backup_16TB/sean/VoxelJEPA/jepa/configs/evals_a6000/vit_base/FT_Bin/FTD_sample/12_1_meanintensity_k400.yaml --devices cuda:0 --validation True --plotter csv"
    # "conda activate jepa && cd /media/backup_16TB/sean/VoxelJEPA/jepa && python -m evals.main --fname /media/backup_16TB/sean/VoxelJEPA/jepa/configs/evals_a6000/vit_base/FT_Bin/FTD_sample/12_1_meanintensity_no_ent.yaml --devices cuda:0 --validation True --plotter csv"
    # "conda activate jepa && cd /media/backup_16TB/sean/VoxelJEPA/jepa && python -m evals.main --fname /media/backup_16TB/sean/VoxelJEPA/jepa/configs/evals_a6000/vit_base/FT_Bin_eval/FTD_sample/12_1_meanintensity_k400.yaml --devices cuda:0 --validation True --plotter csv"
    # "conda activate jepa && cd /media/backup_16TB/sean/VoxelJEPA/jepa && python -m evals.main --fname /media/backup_16TB/sean/VoxelJEPA/jepa/configs/evals_a6000/vit_base/FT_Bin_eval/FTD_sample/12_1_meanintensity_no_ent.yaml --devices cuda:0 --validation True --plotter csv"
    # RAN END
    
    # "conda activate jepa && cd /media/backup_16TB/sean/VoxelJEPA/jepa && python -m evals.main --fname /media/backup_16TB/sean/VoxelJEPA/jepa/configs/evals_a6000/vit_base/FT_Bin/FTD_sample/12_1_meanintensity.yaml --devices cuda:0 --validation True --plotter csv"
    "conda activate jepa && cd /media/backup_16TB/sean/VoxelJEPA/jepa && python -m evals.main --fname /media/backup_16TB/sean/VoxelJEPA/jepa/configs/evals_a6000/vit_base/FT_Bin_eval/FTD_sample/12_1_meanintensity.yaml --devices cuda:0 --validation True --plotter csv"

    # "conda activate jepa && cd /media/backup_16TB/sean/VoxelJEPA/jepa && python -m evals.main --fname /media/backup_16TB/sean/VoxelJEPA/jepa/configs/evals_a6000/vit_base/FT_Bin/allscantrain_singletestval/12_1_meanintensity.yaml --devices cuda:1 --validation True --plotter csv"
    # "conda activate jepa && cd /media/backup_16TB/sean/VoxelJEPA/jepa && python -m evals.main --fname /media/backup_16TB/sean/VoxelJEPA/jepa/configs/evals_a6000/vit_base/FT_Bin/allscantrain_singletestval/12_1_std.yaml --devices cuda:1 --validation True --plotter csv"

    # "conda activate jepa && cd /media/backup_16TB/sean/VoxelJEPA/jepa && python -m evals.main --fname /media/backup_16TB/sean/VoxelJEPA/jepa/configs/evals_a6000/vit_base/FT_Bin_eval/allscantrain_singletestval/12_1_meanintensity.yaml --devices cuda:1 --validation True --plotter csv"
    # "conda activate jepa && cd /media/backup_16TB/sean/VoxelJEPA/jepa && python -m evals.main --fname /media/backup_16TB/sean/VoxelJEPA/jepa/configs/evals_a6000/vit_base/FT_Bin_eval/allscantrain_singletestval/12_1_std.yaml --devices cuda:1 --validation True --plotter csv"

)

# Loop through the commands and run them in sequence
for cmd in "${commands[@]}"; do
    echo "Running: $cmd"
    eval "$cmd"
done