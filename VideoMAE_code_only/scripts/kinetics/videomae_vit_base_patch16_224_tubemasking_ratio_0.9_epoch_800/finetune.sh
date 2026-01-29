# Set the path to save checkpoints
OUTPUT_DIR='/home/tianze/Code/VideoMAE/checkpoints/vit_base_consecutive_finetune_augment_fixed'
# path to Kinetics set (train.csv/val.csv/test.csv)
DATA_PATH='/home/tianze/Code/VJEPA/jepa_sean/configs/evals_a6000_csv/AD_CN'
# path to pretrain model
# MODEL_PATH='/home/tianze/Code/VideoMAE/checkpoints/vit_base_consecutive/checkpoint-49.pth'
MODEL_PATH='/home/tianze/Code/VideoMAE/checkpoints/vit_base_consecutive_finetune_augment_fixed/checkpoint-199/mp_rank_00_model_states.pt'

# batch_size can be adjusted according to number of GPUs
# this script is for 64 GPUs (8 nodes x 8 GPUs)
deepspeed --num_gpus=4 run_class_finetuning.py \
    --model vit_base_patch16_224 \
    --data_set AD_CN \
    --nb_classes 2 \
    --data_path ${DATA_PATH} \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 8 \
    --num_sample 1 \
    --input_size 224 \
    --short_side_size 224 \
    --save_ckpt_freq 10 \
    --num_frames 16 \
    --sampling_rate 4 \
    --opt adamw \
    --lr 1e-3 \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.05 \
    --epochs 200 \
    --test_num_segment 5 \
    --test_num_crop 3 \
    --reprob 0.0 \
    --aa None \
    --smoothing 0.0 \
    --enable_deepspeed \
    --eval \
    --dist_eval \