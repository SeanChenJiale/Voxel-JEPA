# Set the path to save checkpoints
OUTPUT_DIR='/home/tianze/Code/VideoMAE/checkpoints/vit_base_consecutive'
# Set the path to Kinetics train set. 
DATA_PATH='/home/tianze/DATA_2T/PreProcessedData_box_trial/pretraining/T1/pretraining_nii_onlyT1_11_10.csv'

# batch_size can be adjusted according to number of GPUs
# this script is for 64 GPUs (8 nodes x 8 GPUs)
python run_mae_pretraining.py \
        --data_path ${DATA_PATH} \
        --mask_type tube \
        --mask_ratio 0.9 \
        --model pretrain_videomae_base_patch16_224 \
        --decoder_depth 4 \
        --batch_size 24 \
        --num_frames 16 \
        --sampling_rate 4 \
        --opt adamw \
        --opt_betas 0.9 0.95 \
        --warmup_epochs 3 \
        --save_ckpt_freq 10 \
        --epochs 50 \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR}