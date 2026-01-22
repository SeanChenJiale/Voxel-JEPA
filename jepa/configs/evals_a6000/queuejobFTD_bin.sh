#!/bin/bash

# List of PIDs to monitor
PIDS=(3822067 3822066 3822068)

# Function to check if any of the PIDs are still running
are_pids_running() {
    for pid in "${PIDS[@]}"; do
        if ps -p $pid > /dev/null 2>&1; then
            return 0  # At least one PID is still running
        fi
    done
    return 1  # All PIDs have stopped
}

# Wait until all PIDs have stopped
echo "Waiting for PIDs ${PIDS[*]} to stop..."
while are_pids_running; do
    sleep 10  # Check every 10 seconds
done

echo "All PIDs have stopped. Starting the job..."

# Initialize Conda (ensure this points to your Conda installation)
source ~/anaconda3/etc/profile.d/conda.sh

# Activate the Conda environment and navigate to the project directory
conda activate jepa
cd /media/backup_16TB/sean/VoxelJEPA/jepa

# Define the commands to run in each screen
commands=(
    # "conda activate jepa && cd /media/backup_16TB/sean/VoxelJEPA/jepa && python -m evals.main --fname /media/backup_16TB/sean/VoxelJEPA/jepa/configs/evals_a6000/vit_base/FT_Bin/loss/12_1_meanintensity_k400.yaml --devices cuda:1 --validation True --plotter csv"
    # "conda activate jepa && cd /media/backup_16TB/sean/VoxelJEPA/jepa && python -m evals.main --fname /media/backup_16TB/sean/VoxelJEPA/jepa/configs/evals_a6000/vit_base/FT_Bin/loss/12_1_meanintensity_no_ent.yaml --devices cuda:1 --validation True --plotter csv"
    # "conda activate jepa && cd /media/backup_16TB/sean/VoxelJEPA/jepa && python -m evals.main --fname /media/backup_16TB/sean/VoxelJEPA/jepa/configs/evals_a6000/vit_base/FT_Bin/loss/12_1_meanintensity.yaml --devices cuda:1 --validation True --plotter csv"
    # "conda activate jepa && cd /media/backup_16TB/sean/VoxelJEPA/jepa && python -m evals.main --fname /media/backup_16TB/sean/VoxelJEPA/jepa/configs/evals_a6000/vit_base/FT_Bin/loss/12_1_std.yaml --devices cuda:1 --validation True --plotter csv"

    # "conda activate jepa && cd /media/backup_16TB/sean/VoxelJEPA/jepa && python -m evals.main --fname /media/backup_16TB/sean/VoxelJEPA/jepa/configs/evals_a6000/vit_base/FT_Bin_eval/loss/12_1_meanintensity_k400.yaml --devices cuda:1 --validation True --plotter csv"
    # "conda activate jepa && cd /media/backup_16TB/sean/VoxelJEPA/jepa && python -m evals.main --fname /media/backup_16TB/sean/VoxelJEPA/jepa/configs/evals_a6000/vit_base/FT_Bin_eval/loss/12_1_meanintensity_no_ent.yaml --devices cuda:1 --validation True --plotter csv"
    # "conda activate jepa && cd /media/backup_16TB/sean/VoxelJEPA/jepa && python -m evals.main --fname /media/backup_16TB/sean/VoxelJEPA/jepa/configs/evals_a6000/vit_base/FT_Bin_eval/loss/12_1_meanintensity.yaml --devices cuda:1 --validation True --plotter csv"
    # "conda activate jepa && cd /media/backup_16TB/sean/VoxelJEPA/jepa && python -m evals.main --fname /media/backup_16TB/sean/VoxelJEPA/jepa/configs/evals_a6000/vit_base/FT_Bin_eval/loss/12_1_std.yaml --devices cuda:1 --validation True --plotter csv"
    
      #combin loss
    "conda activate jepa && cd /media/backup_16TB/sean/VoxelJEPA/jepa && python -m evals.main --fname /media/backup_16TB/sean/VoxelJEPA/jepa/configs/evals_a6000/vit_base/FT_Bin/loss/12_1_std005_mean01.yaml --devices cuda:1"
    "conda activate jepa && cd /media/backup_16TB/sean/VoxelJEPA/jepa && python -m evals.main --fname /media/backup_16TB/sean/VoxelJEPA/jepa/configs/evals_a6000/vit_base/FT_Bin/loss/12_1_std005_mean01.yaml --devices cuda:1 --validation True --plotter csv"
    "conda activate jepa && cd /media/backup_16TB/sean/VoxelJEPA/jepa && python -m evals.main --fname /media/backup_16TB/sean/VoxelJEPA/jepa/configs/evals_a6000/vit_base/FT_Bin_eval/loss/12_1_std005_mean01.yaml --devices cuda:1 --validation True --plotter csv"

    #   #combin loss diag
    # "conda activate jepa && cd /media/backup_16TB/sean/VoxelJEPA/jepa && python -m evals.main --fname /media/backup_16TB/sean/VoxelJEPA/jepa/configs/evals_a6000/vit_base/FT_Bin/loss/12_1_std03_mean06_diag.yaml --devices cuda:1"
    # "conda activate jepa && cd /media/backup_16TB/sean/VoxelJEPA/jepa && python -m evals.main --fname /media/backup_16TB/sean/VoxelJEPA/jepa/configs/evals_a6000/vit_base/FT_Bin/loss/12_1_std03_mean06_diag.yaml --devices cuda:1 --validation True --plotter csv"
    # "conda activate jepa && cd /media/backup_16TB/sean/VoxelJEPA/jepa && python -m evals.main --fname /media/backup_16TB/sean/VoxelJEPA/jepa/configs/evals_a6000/vit_base/FT_Bin_eval/loss/12_1_std03_mean06_diag.yaml --devices cuda:1 --validation True --plotter csv"

)

# Loop through the commands and run them in sequence
for cmd in "${commands[@]}"; do
    echo "Running: $cmd"
    eval "$cmd"
done