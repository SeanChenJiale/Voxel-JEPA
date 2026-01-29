# Set the path to save video
OUTPUT_DIR='/home/tianze/Code/VideoMAE/visualization/vit_base-ad_sampling-half_mask-cov_loss_entropy-K400-MRI-pretrain/'
# path to video for visualization
VIDEO_PATH='/home/tianze/DATA_2T/MRI/subvolume_pretrain/PPMI_nii_T1/58111_Anon_20230207000000_2_indices_axis0.nii.gz'
# path to pretrain model
MODEL_PATH='/home/tianze/Code/VideoMAE/checkpoints/vit_base-ad_sampling-half_mask-cov_loss_entropy-K400-MRI-pretrain/checkpoint-50.pth'

python3 run_videomae_vis.py \
    --mask_ratio 0.9 \
    --mask_type half \
    --decoder_depth 4 \
    --model pretrain_videomae_base_patch16_224 \
    ${VIDEO_PATH} ${OUTPUT_DIR} ${MODEL_PATH}