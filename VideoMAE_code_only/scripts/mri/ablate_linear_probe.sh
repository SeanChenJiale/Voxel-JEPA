#!/bin/bash

# Navigate to project root
# cd "$(dirname "$0")/../.."

# Configuration
DATA_PATH='/home/tianze/DATA_2T/MRI/csv_for_finetuning/Finalized_all/Downstreamtask_csv'

# Checkpoints
# Using the single checkpoint requested by user
CKPT='/home/tianze/Code/VideoMAE/checkpoints/final_experiments/K400-MRI-cov_loss_mean_diag_weight_1-pretrain/checkpoint-50.pth'
ckpt_name="cov_loss_mean_diag_weight_1"

datasets=("MCI_CN" "AD_MCI_CN" "NIFD" "IXI_gender")
percentages=(100 75 50 25)

# Function to get nb_classes
get_nb_classes() {
    if [ "$1" == "AD_MCI_CN" ]; then
        echo 3
    elif [ "$1" == "IXI_age" ]; then
        echo 1
    else
        echo 2
    fi
}

# Base output directory
BASE_OUTPUT_DIR='/home/tianze/Code/VideoMAE/checkpoints/ablation_linear_probe'

# Create a list of all tasks
tasks=()
for ds in "${datasets[@]}"; do
    for pct in "${percentages[@]}"; do
        tasks+=("$ckpt_name|$CKPT|$ds|$pct")
    done
done

# Total tasks: 1 * 4 * 4 = 16
total_tasks=${#tasks[@]}
available_gpus=(1 2 3)
num_gpus=${#available_gpus[@]}
tasks_per_gpu=$(( (total_tasks + num_gpus - 1) / num_gpus ))

run_task() {
    local task_info=$1
    local gpu_id=$2
    
    IFS='|' read -r c_name ckpt_path ds pct <<< "$task_info"
    
    nb_classes=$(get_nb_classes "$ds")
    
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/${c_name}/${ds}_pct${pct}"
    mkdir -p ${OUTPUT_DIR}
    
    echo "Starting: Ckpt=$c_name, Dataset=$ds, Pct=$pct on cuda:$gpu_id"
    
    python run_class_finetuning.py \
        --model vit_base_patch16_224 \
        --data_set ${ds} \
        --nb_classes ${nb_classes} \
        --device cuda:${gpu_id} \
        --data_path ${DATA_PATH} \
        --train_pct ${pct} \
        --finetune ${ckpt_path} \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --batch_size 32 \
        --num_sample 1 \
        --input_size 224 \
        --short_side_size 224 \
        --save_ckpt_freq 50 \
        --num_frames 16 \
        --sampling_rate 4 \
        --opt adamw \
        --lr 0.01 \
        --opt_betas 0.9 0.999 \
        --weight_decay 0.001 \
        --layer_decay 1.0 \
        --epochs 200 \
        --test_num_segment 5 \
        --test_num_crop 3 \
        --reprob 0.0 \
        --aa None \
        --smoothing 0.0 \
        --dist_eval \
        --linear_probe \
        > ${OUTPUT_DIR}/train.log 2>&1
    
    echo "Finished: Ckpt=$c_name, Dataset=$ds, Pct=$pct on cuda:$gpu_id"
}

# Distribute tasks across available GPUs (1 to 3)
for ((g=0; g<num_gpus; g++)); do
    gpu_id=${available_gpus[$g]}
    (
        start_idx=$(( g * tasks_per_gpu ))
        end_idx=$(( (g + 1) * tasks_per_gpu ))
        if [ $end_idx -gt $total_tasks ]; then
            end_idx=$total_tasks
        fi
        
        for ((i=start_idx; i<end_idx; i++)); do
            run_task "${tasks[$i]}" "$gpu_id"
        done
    ) &
done

echo "All $total_tasks tasks started in background on cuda:1 to cuda:3. Check ${BASE_OUTPUT_DIR} for logs."
wait
