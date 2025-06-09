from evals.video_regression_frozen import eval
import torch
import src.models.vision_transformer as vit
import yaml
import logging

def validate_on_video(video_path, pretrained_model_path, args_eval):
    device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')

    # Load the pretrained model
    encoder = init_model(
        device=device,
        pretrained=pretrained_model_path,
        model_name=args_eval['pretrain']['model_name'],
        patch_size=args_eval['pretrain']['patch_size'],
        crop_size=args_eval['optimization']['resolution'],
        frames_per_clip=args_eval['data']['frames_per_clip'],
        tubelet_size=args_eval['pretrain']['tubelet_size'],
        use_sdpa=args_eval['pretrain']['use_sdpa'],
        use_SiLU=args_eval['pretrain']['use_silu'],
        tight_SiLU=args_eval['pretrain']['tight_silu'],
        uniform_power=args_eval['pretrain']['uniform_power'],
        checkpoint_key=args_eval['pretrain']['checkpoint_key']
    )
    encoder.eval()

    # Initialize the classifier
    classifier = AttentiveClassifier(
        embed_dim=encoder.embed_dim,
        num_heads=encoder.num_heads,
        depth=1,
        num_classes=args_eval['data']['num_classes']
    ).to(device)
    classifier.eval()

    # Create a DataLoader for the video
    video_loader = make_dataloader(
        root_path=[video_path],
        batch_size=1,  # Single video
        world_size=1,
        rank=0,
        dataset_type=args_eval['data']['dataset_type'],
        resolution=args_eval['optimization']['resolution'],
        frames_per_clip=args_eval['data']['frames_per_clip'],
        frame_step=args_eval['pretrain']['frame_step'],
        num_segments=args_eval['data']['num_segments'],
        eval_duration=args_eval['pretrain']['clip_duration'],
        num_views_per_segment=args_eval['data']['num_views_per_segment'],
        allow_segment_overlap=True,
        training=False
    )

    # Run validation
    val_acc = run_one_epoch(
        device=device,
        training=False,
        encoder=encoder,
        classifier=classifier,
        scaler=None,  # No gradient scaling needed for validation
        optimizer=None,  # Not needed for validation
        scheduler=None,  # Not needed for validation
        wd_scheduler=None,  # Not needed for validation
        data_loader=video_loader,
        use_bfloat16=args_eval['optimization']['use_bfloat16'],
        num_spatial_views=1,
        num_temporal_views=args_eval['data']['num_segments'],
        attend_across_segments=args_eval['optimization']['attend_across_segments']
    )

    print(f'Validation Accuracy: {val_acc:.2f}%')

# args_eval = read_yaml('/local_data/sean_hasitha/sean/VJEPA/jepa/configs/evals/vitl16_brainslice48_axisclassifier_scratch.yaml')


# pretraining_folder = args_eval['pretrain']['folder']
# eval_name = args_eval['eval_name']
# tag = args_eval['tag']
# write_tag = args_eval['pretrain']['write_tag']

# pretrained_model_path = f'{pretraining_folder}/{eval_name}/{tag}/{write_tag}-latest.pth.tar'

# validate_on_video('/local_data/sean_hasitha/sean/Dataset_downstream48/2/val/SOOP/sub-13_T1w/sub-13_T1w/Sagittal.mp4',
#                   pretrained_model_path,
#                   args_eval)