#!/bin/bash

# List of PIDs to monitor
PIDS=(3583476)

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

# Run the Python script
python -m app.main --fname /media/backup_16TB/sean/VoxelJEPA/jepa/configs/pretrain_a6000/vit_base/entropy_loss/12_1_2026/decoder_mean_0_1_std_0_05.yaml --devices cuda:0 cuda:1 cuda:2
