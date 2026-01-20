#!/bin/bash

echo "All PIDs have stopped. Starting the job..."

# Initialize Conda (ensure this points to your Conda installation)
source ~/anaconda3/etc/profile.d/conda.sh

# Activate the Conda environment and navigate to the project directory
conda activate jepa
cd /media/backup_16TB/sean/VoxelJEPA/jepa

# Run the Python script
python -m app.main --fname /media/backup_16TB/sean/VoxelJEPA/jepa/configs/pretrain_a6000/vit_base/entropy_loss/12_1_2026/decoder_mean_std.yaml --devices cuda:0 cuda:1 cuda:2
