#!/bin/bash

# Navigate to project root
# cd "$(dirname "$0")/../.."

# Configuration
DATA_PATH='/home/tianze/DATA_2T/MRI/csv_for_finetuning/Finalized_all/Downstreamtask_csv'

# Checkpoints
CKPT1='/home/tianze/Code/VideoMAE/checkpoints/vit_base_K400_pretrain_800epochs/checkpoint.pth'
CKPT2='/home/tianze/Code/VideoMAE/checkpoints/final_experiments/K400-MRI-cov_loss_comb_diag-pretrain/checkpoint-50.pth'

checkpoints=("$CKPT1" "$CKPT2")
ckpt_names=("K400_orig" "cov_loss_comb_diag")

# Dataset
DATASET='IXI_age'
NUM_CLASSES=1  # Regression task for age prediction

# Base output directory
BASE_OUTPUT_DIR='/home/tianze/Code/VideoMAE/checkpoints/finetune_IXI_age'

# Create a list of all tasks
tasks=()
for i in "${!checkpoints[@]}"; do
    tasks+=("${ckpt_names[$i]}|${checkpoints[$i]}")
done

# Total tasks: 2
total_tasks=${#tasks[@]}

run_task() {
    local task_info=$1
    local gpu_id=$2
    
    IFS='|' read -r c_name ckpt_path <<< "$task_info"
    
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/${c_name}"
    mkdir -p ${OUTPUT_DIR}
    
    echo "Starting: Ckpt=$c_name on cuda:$gpu_id"
    
    python run_class_finetuning.py \
        --model vit_base_patch16_224 \
        --data_set ${DATASET} \
        --nb_classes ${NUM_CLASSES} \
        --device cuda:${gpu_id} \
        --data_path ${DATA_PATH} \
        --finetune ${ckpt_path} \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --batch_size 6 \
        --num_sample 1 \
        --input_size 224 \
        --short_side_size 224 \
        --save_ckpt_freq 20 \
        --num_frames 16 \
        --sampling_rate 4 \
        --opt adamw \
        --lr 1e-3 \
        --opt_betas 0.9 0.999 \
        --weight_decay 0.05 \
        --epochs 100 \
        --test_num_segment 5 \
        --test_num_crop 3 \
        --reprob 0.0 \
        --aa None \
        --smoothing 0.0 \
        --dist_eval \
        --layer_decay 0.5 \
        > ${OUTPUT_DIR}/train.log 2>&1
    
    echo "Finished: Ckpt=$c_name on cuda:$gpu_id"
}

# Run tasks sequentially on cuda:0
for task in "${tasks[@]}"; do
    run_task "$task" 0
done

echo "All $total_tasks tasks completed on cuda:0. Check ${BASE_OUTPUT_DIR} for logs."
