# Set the path to save checkpoints
OUTPUT_DIR='/home/tianze/Code/VideoMAE/checkpoints/final_experiments/MRI-cov_loss_comb_diag_weight_0.5-pretrain/'
# OUTPUT_DIR='/home/tianze/Code/VideoMAE/checkpoints/vit_base-ad_sampling-half_mask-no_normalize-cov_loss-K400-MRI-pretrain/'
# Set the path to Kinetics train set. 
DATA_PATH='/home/tianze/DATA_2T/MRI/subvolume_pretrain/box_indices_filtered_no_adni_final_AD_strat.csv'
# Set the path to pretrain model
# MODEL_PATH='/home/tianze/Code/VideoMAE/checkpoints/vit_base_K400_pretrain_800epochs/checkpoint.pth'
MODEL_PATH=None


# batch_size can be adjusted according to number of GPUs
# this script is for 64 GPUs (8 nodes x 8 GPUs)
mkdir -p ${OUTPUT_DIR}
nohup python run_mae_pretraining.py \
        --data_path ${DATA_PATH} \
        --device cuda:1 \
        --mask_type half \
        --mask_ratio 0.9 \
        --model pretrain_videomae_base_patch16_224 \
        --decoder_depth 4 \
        --batch_size 12 \
        --num_frames 16 \
        --sampling_rate 4 \
        --opt adamw \
        --opt_betas 0.9 0.95 \
        --warmup_epochs 3 \
        --save_ckpt_freq 15 \
        --epochs 51 \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --normlize_target True \
        --use_cov_loss \
        --cov_loss_weight 0.5 \
        --feature_type mean_std \
        --use_trace_sum False \
        > ${OUTPUT_DIR}train.log 2>&1 &
        # REMEMBER to add --checkpoint ${MODEL_PATH} 