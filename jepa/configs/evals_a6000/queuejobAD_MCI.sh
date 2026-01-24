#!/bin/bash

# Initialize Conda (ensure this points to your Conda installation)
source ~/anaconda3/etc/profile.d/conda.sh

# "conda activate jepa && cd /media/backup_16TB/sean/VoxelJEPA/jepa && python -m evals.main --fname /media/backup_16TB/sean/VoxelJEPA/jepa/configs/evals_a6000/vit_base/MCI_CN/std_loss/12_1_std.yaml --devices cuda:0"
# "conda activate jepa && cd /media/backup_16TB/sean/VoxelJEPA/jepa && python -m evals.main --fname /media/backup_16TB/sean/VoxelJEPA/jepa/configs/evals_a6000/vit_base/ADCNMCI/std_loss/12_1_std.yaml --devices cuda:0"

# Define the commands to run in each screen
commands=(
#combin loss

"conda activate jepa && cd /media/backup_16TB/sean/VoxelJEPA/jepa && python -m evals.main --fname /media/backup_16TB/sean/VoxelJEPA/jepa/configs/evals_a6000/vit_base/MCI_CN_ablation_25/combin_loss/12_1_0_3std_0_6_mean_diag.yaml --devices cuda:0"
"conda activate jepa && cd /media/backup_16TB/sean/VoxelJEPA/jepa && python -m evals.main --fname /media/backup_16TB/sean/VoxelJEPA/jepa/configs/evals_a6000/vit_base/MCI_CN_ablation_25/combin_loss/12_1_0_3std_0_6_mean_diag.yaml --devices cuda:0 --validation True --plotter csv"
"conda activate jepa && cd /media/backup_16TB/sean/VoxelJEPA/jepa && python -m evals.main --fname /media/backup_16TB/sean/VoxelJEPA/jepa/configs/evals_a6000/vit_base/MCI_CN_ablation_25_eval/combin_loss/12_1_0_3std_0_6_mean_diag.yaml --devices cuda:0 --validation True --plotter csv"

"conda activate jepa && cd /media/backup_16TB/sean/VoxelJEPA/jepa && python -m evals.main --fname /media/backup_16TB/sean/VoxelJEPA/jepa/configs/evals_a6000/vit_base/MCI_CN_ablation_25/combin_loss/12_1_0_05std_0_1_mean_trace.yaml --devices cuda:0"
"conda activate jepa && cd /media/backup_16TB/sean/VoxelJEPA/jepa && python -m evals.main --fname /media/backup_16TB/sean/VoxelJEPA/jepa/configs/evals_a6000/vit_base/MCI_CN_ablation_25/combin_loss/12_1_0_05std_0_1_mean_trace.yaml --devices cuda:0 --validation True --plotter csv"
"conda activate jepa && cd /media/backup_16TB/sean/VoxelJEPA/jepa && python -m evals.main --fname /media/backup_16TB/sean/VoxelJEPA/jepa/configs/evals_a6000/vit_base/MCI_CN_ablation_25_eval/combin_loss/12_1_0_05std_0_1_mean_trace.yaml --devices cuda:0 --validation True --plotter csv"

#mean int

"conda activate jepa && cd /media/backup_16TB/sean/VoxelJEPA/jepa && python -m evals.main --fname /media/backup_16TB/sean/VoxelJEPA/jepa/configs/evals_a6000/vit_base/MCI_CN_ablation_25/mean_loss/12_1_meanintensity_k400.yaml --devices cuda:0"
"conda activate jepa && cd /media/backup_16TB/sean/VoxelJEPA/jepa && python -m evals.main --fname /media/backup_16TB/sean/VoxelJEPA/jepa/configs/evals_a6000/vit_base/MCI_CN_ablation_25/mean_loss/12_1_meanintensity_k400.yaml --devices cuda:0 --validation True --plotter csv"
"conda activate jepa && cd /media/backup_16TB/sean/VoxelJEPA/jepa && python -m evals.main --fname /media/backup_16TB/sean/VoxelJEPA/jepa/configs/evals_a6000/vit_base/MCI_CN_ablation_25_eval/mean_loss/12_1_meanintensity_k400.yaml --devices cuda:0 --validation True --plotter csv"

"conda activate jepa && cd /media/backup_16TB/sean/VoxelJEPA/jepa && python -m evals.main --fname /media/backup_16TB/sean/VoxelJEPA/jepa/configs/evals_a6000/vit_base/MCI_CN_ablation_25/mean_loss/12_1_meanintensity.yaml --devices cuda:0"
"conda activate jepa && cd /media/backup_16TB/sean/VoxelJEPA/jepa && python -m evals.main --fname /media/backup_16TB/sean/VoxelJEPA/jepa/configs/evals_a6000/vit_base/MCI_CN_ablation_25/mean_loss/12_1_meanintensity.yaml --devices cuda:0 --validation True --plotter csv"
"conda activate jepa && cd /media/backup_16TB/sean/VoxelJEPA/jepa && python -m evals.main --fname /media/backup_16TB/sean/VoxelJEPA/jepa/configs/evals_a6000/vit_base/MCI_CN_ablation_25_eval/mean_loss/12_1_meanintensity.yaml --devices cuda:0 --validation True --plotter csv"

"conda activate jepa && cd /media/backup_16TB/sean/VoxelJEPA/jepa && python -m evals.main --fname /media/backup_16TB/sean/VoxelJEPA/jepa/configs/evals_a6000/vit_base/MCI_CN_ablation_25/mean_loss/12_1_no_meanintensity.yaml --devices cuda:0"
"conda activate jepa && cd /media/backup_16TB/sean/VoxelJEPA/jepa && python -m evals.main --fname /media/backup_16TB/sean/VoxelJEPA/jepa/configs/evals_a6000/vit_base/MCI_CN_ablation_25/mean_loss/12_1_no_meanintensity.yaml --devices cuda:0 --validation True --plotter csv"
"conda activate jepa && cd /media/backup_16TB/sean/VoxelJEPA/jepa && python -m evals.main --fname /media/backup_16TB/sean/VoxelJEPA/jepa/configs/evals_a6000/vit_base/MCI_CN_ablation_25_eval/mean_loss/12_1_no_meanintensity.yaml --devices cuda:0 --validation True --plotter csv"

#std dev

"conda activate jepa && cd /media/backup_16TB/sean/VoxelJEPA/jepa && python -m evals.main --fname /media/backup_16TB/sean/VoxelJEPA/jepa/configs/evals_a6000/vit_base/MCI_CN_ablation_25/std_loss/12_1_std.yaml --devices cuda:0"
"conda activate jepa && cd /media/backup_16TB/sean/VoxelJEPA/jepa && python -m evals.main --fname /media/backup_16TB/sean/VoxelJEPA/jepa/configs/evals_a6000/vit_base/MCI_CN_ablation_25/std_loss/12_1_std.yaml --devices cuda:0 --validation True --plotter csv"
"conda activate jepa && cd /media/backup_16TB/sean/VoxelJEPA/jepa && python -m evals.main --fname /media/backup_16TB/sean/VoxelJEPA/jepa/configs/evals_a6000/vit_base/MCI_CN_ablation_25_eval/std_loss/12_1_std.yaml --devices cuda:0 --validation True --plotter csv"

)

# Loop through the commands and run them in sequence
for cmd in "${commands[@]}"; do
    echo "Running: $cmd"
    eval "$cmd"
done