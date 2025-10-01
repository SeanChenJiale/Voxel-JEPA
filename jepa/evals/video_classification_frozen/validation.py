# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os

# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
try:
    # -- WARNING: IF DOING DISTRIBUTED TRAINING ON A NON-SLURM CLUSTER, MAKE
    # --          SURE TO UPDATE THIS TO GET LOCAL-RANK ON NODE, OR ENSURE
    # --          THAT YOUR JOBS ARE LAUNCHED WITH ONLY 1 DEVICE VISIBLE
    # --          TO EACH PROCESS
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['SLURM_LOCALID']
except Exception:
    pass

import logging
import pprint

import numpy as np

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F

from torch.nn.parallel import DistributedDataParallel

import src.models.vision_transformer as vit
from src.models.attentive_pooler import AttentiveClassifier
from src.datasets.data_manager import (
    init_data,
)
from src.utils.distributed import (
    init_distributed,
    AllReduce
)
from src.utils.schedulers import (
    WarmupCosineSchedule,
    CosineWDSchedule,
)
from src.utils.logging import (
    AverageMeter,
    CSVLogger,
    init_csv_writer
)

from evals.video_classification_frozen.utils import (
    make_transforms,
    ClipAggregation,
    FrameAggregation
)

from evals.video_classification_frozen.confusion_matrix import (
    plot_confusion_matrix
)

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

pp = pprint.PrettyPrinter(indent=4)

def load_classifier_weights(model, checkpoint_path, key='classifier', strict=False):
    """
    Load classifier weights into the given model.

    Args:
        model (torch.nn.Module): The model to load the weights into.
        checkpoint_path (str): Path to the checkpoint file.
        key (str): Key in the checkpoint containing the state dictionary.
        strict (bool): Whether to strictly enforce that the keys in `state_dict` match the model's keys.

    Returns:
        tuple: A tuple containing lists of missing keys and unexpected keys.
    """
    try:
        # Load the checkpoint
        logger.info(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Extract the state dictionary for the classifier
        if key not in checkpoint:
            raise KeyError(f"Key '{key}' not found in checkpoint. Available keys: {list(checkpoint.keys())}")
        state_dict = checkpoint[key]

        # Remove "module." prefix from keys if present (for DataParallel/DistributedDataParallel models)
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        # Load the state dictionary into the model
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)

        # Log the results
        if missing_keys:
            logger.warning(f"Missing keys when loading state_dict: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys when loading state_dict: {unexpected_keys}")
        logger.info(f"Successfully loaded weights from checkpoint: {checkpoint_path}")

        return missing_keys, unexpected_keys

    except FileNotFoundError:
        logger.error(f"Checkpoint file not found: {checkpoint_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading checkpoint: {e}")
        raise


def main(args_eval, plotter, resume_preempt=False, debug=False):
    print("\n\n\n\n IN VALIDATION MODE \n\n\n\n")
    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- PRETRAIN
    args_eval_name = args_eval.get('eval_name')
    args_pretrain = args_eval.get('pretrain')
    checkpoint_key = args_pretrain.get('checkpoint_key', 'target_encoder')
    model_name = args_pretrain.get('model_name', None)
    patch_size = args_pretrain.get('patch_size', None)
    pretrain_folder = args_pretrain.get('folder', None)
    ckp_fname = args_pretrain.get('checkpoint', None)
    tag = args_pretrain.get('write_tag', None)
    use_sdpa = args_pretrain.get('use_sdpa', True)
    use_SiLU = args_pretrain.get('use_silu', False)
    tight_SiLU = args_pretrain.get('tight_silu', True)
    uniform_power = args_pretrain.get('uniform_power', False)
    strategy = args_pretrain.get('strategy','consecutive') # default mri selection is 'consecutive' other is 'skip_1'
    pretrained_path = os.path.join(pretrain_folder, ckp_fname)
    # Optional [for Video model]:
    tubelet_size = args_pretrain.get('tubelet_size', 2)
    pretrain_frames_per_clip = args_pretrain.get('frames_per_clip', 1)

    # -- DATA
    args_data = args_eval.get('data')
    train_data_path = [args_data.get('dataset_train')]
    val_data_path = [args_data.get('dataset_val')]
    dataset_type = args_data.get('dataset_type', 'VideoDataset')
    num_classes = args_data.get('num_classes')
    eval_num_segments = args_data.get('num_segments', 1)
    eval_frames_per_clip = args_data.get('frames_per_clip', 16)
    eval_frame_step = args_pretrain.get('frame_step', 4)
    eval_duration = args_pretrain.get('clip_duration', None)
    eval_num_views_per_segment = args_data.get('num_views_per_segment', 1)
    if num_classes > 2:
        is_multiclass = True
    else: is_multiclass = False

    # -- DATA AUGS
    cfgs_data_aug = args_eval.get('data_aug',None)
    if cfgs_data_aug is not None:
        ar_range = cfgs_data_aug.get('random_resize_aspect_ratio', [3/4, 4/3])
        rr_scale = cfgs_data_aug.get('random_resize_scale', [0.3, 1.0])
        motion_shift = cfgs_data_aug.get('motion_shift', False)
        reprob = cfgs_data_aug.get('reprob', 0.)
        use_aa = cfgs_data_aug.get('auto_augment', False)
        tensor_normalize = cfgs_data_aug.get('normalize', ((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))) #use Imagenet if nor provided
        data_aug_dict = dict(ar_range=ar_range,
                        rr_scale=rr_scale,
                        motion_shift=motion_shift,
                        tensor_normalize=tensor_normalize)
    else:
        data_aug_dict = dict(ar_range=[3/4, 4/3],
                        rr_scale=[0.3, 1.0],
                        motion_shift=False,
                        tensor_normalize=((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)))
    # -- OPTIMIZATION
    args_opt = args_eval.get('optimization')
    resolution = args_opt.get('resolution', 224)
    batch_size = args_opt.get('batch_size')
    attend_across_segments = args_opt.get('attend_across_segments', False)
    num_epochs = args_opt.get('num_epochs')
    wd = args_opt.get('weight_decay')
    start_lr = args_opt.get('start_lr')
    lr = args_opt.get('lr')
    final_lr = args_opt.get('final_lr')
    warmup = args_opt.get('warmup')
    use_bfloat16 = args_opt.get('use_bfloat16')

    # -- EXPERIMENT-ID/TAG (optional)
    resume_checkpoint = args_eval.get('resume_checkpoint', False) or resume_preempt
    eval_tag = args_eval.get('tag', None)

    # ----------------------------------------------------------------------- #

    try:
        mp.set_start_method('spawn')
    except Exception:
        pass

    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)

    world_size, rank = init_distributed()
    logger.info(f'Initialized (rank/world-size) {rank}/{world_size}')

    # -- log/checkpointing paths
    folder = os.path.join(pretrain_folder, 'video_classification_frozen/')
    if eval_tag is not None:
        folder = os.path.join(folder, eval_tag)
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    log_file = os.path.join(folder, f'{tag}_r{rank}.csv')
    latest_path = os.path.join(folder, f'{tag}-best.pth.tar')
    if not os.path.exists(latest_path):
        print("\n\n\nLOADING FROM LATEST PATH\n\n\n")
        latest_path = os.path.join(folder, f'{tag}-latest.pth.tar')
    else:
        print("\n\n\nLOADING FROM BEST PATH\n\n\n")

    # Initialize model

    # -- pretrained encoder (frozen)
    encoder = init_model(
        crop_size=resolution,
        device=device,
        pretrained=pretrained_path,
        model_name=model_name,
        patch_size=patch_size,
        tubelet_size=tubelet_size,
        frames_per_clip=pretrain_frames_per_clip,
        uniform_power=uniform_power,
        checkpoint_key=checkpoint_key,
        use_SiLU=use_SiLU,
        tight_SiLU=tight_SiLU,
        use_sdpa=use_sdpa)
    if pretrain_frames_per_clip == 1:
        # Process each frame independently and aggregate
        encoder = FrameAggregation(encoder).to(device)
    else:
        # Process each video clip independently and aggregate
        encoder = ClipAggregation(
            encoder,
            tubelet_size=tubelet_size,
            attend_across_segments=attend_across_segments
        ).to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    # -- init classifier
    classifier = AttentiveClassifier(
        embed_dim=encoder.embed_dim,
        num_heads=encoder.num_heads,
        depth=1,
        num_classes=num_classes,
    ).to(device)

    # -- load classifier weights
    load_classifier_weights(
        model=classifier,
        checkpoint_path=latest_path,
        key='classifier',
        strict=False
    )

    val_loader = make_dataloader(
        dataset_type=dataset_type,
        root_path=val_data_path,
        resolution=resolution,
        frames_per_clip=eval_frames_per_clip,
        frame_step=eval_frame_step,
        num_segments=eval_num_segments,
        eval_duration=eval_duration,
        num_views_per_segment=eval_num_views_per_segment,
        allow_segment_overlap=True,
        batch_size=batch_size,
        world_size=world_size,
        rank=rank,
        training=False,
        debug=debug,
        data_aug_dict=data_aug_dict,
        strategy=strategy) #+)


    # TRAIN LOOP
    writer,csv_file = init_csv_writer(os.path.join(pretrain_folder, args_eval_name,eval_tag, f'{tag}eval_results.csv'))
    if dataset_type == 'VideoDataset':
        writer.writerow(["Label", "Prediction", "Fname", "Probabilities"])
    elif dataset_type.lower() == 'mridataset':
        writer.writerow(["Label", "Prediction", "Fname", "Axis", "Probabilities"])
    for epoch in range(1):
        val_acc = run_one_epoch(
            device=device,
            training=False,
            num_temporal_views=eval_num_segments,
            attend_across_segments=attend_across_segments,
            num_spatial_views=eval_num_views_per_segment,
            encoder=encoder,
            classifier=classifier,
            scaler=None, #validation doesnt need
            optimizer=None, #validation doesnt need
            scheduler=None, #validation doesnt need
            wd_scheduler=None, #validation doesnt need
            data_loader=val_loader,
            use_bfloat16=use_bfloat16,
            writer=writer,
            csv_file=csv_file,
            dataset_type=dataset_type,)

        logger.info('[%5d] test: %.3f%%' % (epoch + 1, val_acc))
    
    print(f"Validation results saved to {os.path.join(pretrain_folder, args_eval_name,eval_tag, f'{tag}eval_results.csv')}")
    plot_confusion_matrix(os.path.join(pretrain_folder, args_eval_name,eval_tag, f'{tag}eval_results.csv'),is_multiclass)

    #######################


def run_one_epoch(
    device,
    training,
    encoder,
    classifier,
    scaler,
    optimizer,
    scheduler,
    wd_scheduler,
    data_loader,
    use_bfloat16,
    num_spatial_views,
    num_temporal_views,
    attend_across_segments,
    writer,
    csv_file,
    dataset_type
):
    
    classifier.train(mode=training)
    criterion = torch.nn.CrossEntropyLoss()
    top1_meter = AverageMeter()
    print(f"Number of samples in the dataset: {len(data_loader.dataset)}")
    print(f"Number of batches in the DataLoader: {len(data_loader)}")
    print(f"currently in training mode? {training}")
    all_labels = []
    all_predictions = []
    all_clip_names = []  # To store clip names or identifiers
    all_axis = []  # To store axis information if available
    all_probabilities = []
    for itr, data in enumerate(data_loader):
        
        if training:
            scheduler.step()
            wd_scheduler.step()

        with torch.cuda.amp.autocast(dtype=torch.float16, enabled=use_bfloat16):

            # Load data and put on GPU
            clips = [
                [dij.to(device, non_blocking=True) for dij in di]  # iterate over spatial views of clip
                for di in data[0]  # iterate over temporal index of clip
            ]
            clip_indices = [d.to(device, non_blocking=True) for d in data[2]]
            labels = data[1].to(device)
            batch_size = len(labels)

            # Forward and prediction
            with torch.no_grad():
                outputs = encoder(clips, clip_indices)
                if not training:
                    if attend_across_segments:
                        outputs = [classifier(o) for o in outputs]
                    else:
                        outputs = [[classifier(ost) for ost in os] for os in outputs]
            if training:
                if attend_across_segments:
                    outputs = [classifier(o) for o in outputs]
                else:
                    outputs = [[classifier(ost) for ost in os] for os in outputs]
            # Get predictions
            if attend_across_segments:
                predictions = torch.argmax(sum([F.softmax(o, dim=1) for o in outputs]) / len(outputs), dim=1)
            else:
                predictions = torch.argmax(
                    sum([sum([F.softmax(ost, dim=1) for ost in os]) for os in outputs]) / len(outputs) / len(outputs[0]),
                    dim=1
                )

        # Save labels and predictions

        all_probabilities.extend(outputs[0][0].cpu().numpy())
        all_labels.extend(labels.cpu().numpy().flatten())
        all_predictions.extend(predictions.cpu().numpy())
        all_clip_names.extend(data[3])
        if dataset_type.lower() == 'mridataset':
            all_axis.extend(data[5].cpu().tolist())

        # Compute loss
        if attend_across_segments:
            loss = sum([criterion(o, labels) for o in outputs]) / len(outputs)
        else:
            loss = sum([sum([criterion(ost, labels) for ost in os]) for os in outputs]) / len(outputs) / len(outputs[0])
        with torch.no_grad():
            if attend_across_segments:
                outputs = sum([F.softmax(o, dim=1) for o in outputs]) / len(outputs)
            else:
                outputs = sum([sum([F.softmax(ost, dim=1) for ost in os]) for os in outputs]) / len(outputs) / len(outputs[0])
            top1_acc = 100. * outputs.max(dim=1).indices.eq(labels).sum() / batch_size
            top1_acc = float(AllReduce.apply(top1_acc))
            top1_meter.update(top1_acc)
        


        if training:
            if use_bfloat16:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
                optimizer.step()
            optimizer.zero_grad()


        logger.info('[%5d] %.3f%% (loss: %.3f) [mem: %.2e]'
                    % (itr, top1_meter.avg, loss,
                        torch.cuda.max_memory_allocated() / 1024.**2))
    if dataset_type.lower() == 'mridataset':
        writer.writerows(zip(all_labels, all_predictions,all_clip_names, all_axis, all_probabilities))  # Write rows
    else:
        writer.writerows(zip(all_labels, all_predictions,all_clip_names, all_probabilities))  # Write rows  # Write rows)

    csv_file.close()
    return top1_meter.avg


def load_checkpoint(
    device,
    r_path,
    classifier,
    opt,
    scaler
):
    try:
        checkpoint = torch.load(r_path, map_location=torch.device('cpu'))
        epoch = checkpoint['epoch']

        # -- loading encoder
        pretrained_dict = checkpoint['classifier']
        msg = classifier.load_state_dict(pretrained_dict)
        logger.info(f'loaded pretrained classifier from epoch {epoch} with msg: {msg}')

        # -- loading optimizer
        opt.load_state_dict(checkpoint['opt'])
        if scaler is not None:
            scaler.load_state_dict(checkpoint['scaler'])
        logger.info(f'loaded optimizers from epoch {epoch}')
        logger.info(f'read-path: {r_path}')
        del checkpoint

    except Exception as e:
        logger.info(f'Encountered exception when loading checkpoint {e}')
        epoch = 0

    return classifier, opt, scaler, epoch


def load_pretrained(
    encoder,
    pretrained,
    checkpoint_key='target_encoder'
):
    logger.info(f'Loading pretrained model from {pretrained}')
    checkpoint = torch.load(pretrained, map_location='cpu')
    try:
        pretrained_dict = checkpoint[checkpoint_key]
    except Exception:
        pretrained_dict = checkpoint['encoder']

    pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items()}
    pretrained_dict = {k.replace('backbone.', ''): v for k, v in pretrained_dict.items()}
    for k, v in encoder.state_dict().items():
        if k not in pretrained_dict:
            logger.info(f'key "{k}" could not be found in loaded state dict')
        elif pretrained_dict[k].shape != v.shape:
            logger.info(f'key "{k}" is of different shape in model and loaded state dict')
            pretrained_dict[k] = v
    msg = encoder.load_state_dict(pretrained_dict, strict=False)
    print(encoder)
    logger.info(f'loaded pretrained model with msg: {msg}')
    logger.info(f'loaded pretrained encoder from epoch: {checkpoint["epoch"]}\n path: {pretrained}')
    del checkpoint
    return encoder


def make_dataloader(
    root_path,
    batch_size,
    world_size,
    rank,
    data_aug_dict,
    dataset_type='VideoDataset',
    resolution=224,
    frames_per_clip=16,
    frame_step=4,
    num_segments=8,
    eval_duration=None,
    num_views_per_segment=1,
    allow_segment_overlap=True,
    training=False,
    num_workers=12,
    subset_file=None,
    debug=False,
    strategy='consecutive' #+
):
    ar_range = data_aug_dict['ar_range']
    rr_scale = data_aug_dict['rr_scale']
    tensor_normalize = data_aug_dict['tensor_normalize']
    # Make Video Transforms
    transform = make_transforms(
        training=training,
        num_views_per_clip=num_views_per_segment,
        random_horizontal_flip=False,
        random_resize_aspect_ratio=None,
        random_resize_scale=None,
        reprob=0.25,
        auto_augment=True,
        motion_shift=False,
        crop_size=resolution,
        normalize=tensor_normalize
    )

    data_loader, _ = init_data(
        data=dataset_type,
        root_path=root_path,
        transform=transform,
        batch_size=batch_size,
        world_size=world_size,
        rank=rank,
        clip_len=frames_per_clip,
        frame_sample_rate=frame_step,
        duration=eval_duration,
        num_clips=num_segments,
        allow_clip_overlap=allow_segment_overlap,
        num_workers=num_workers,
        copy_data=False,
        drop_last=False,
        subset_file=subset_file,
        debug=debug,
        strategy=strategy,) #+)
    return data_loader


def init_model(
    device,
    pretrained,
    model_name,
    patch_size=16,
    crop_size=224,
    # Video specific parameters
    frames_per_clip=16,
    tubelet_size=2,
    use_sdpa=False,
    use_SiLU=False,
    tight_SiLU=True,
    uniform_power=False,
    checkpoint_key='target_encoder'
):
    encoder = vit.__dict__[model_name](
        img_size=crop_size,
        patch_size=patch_size,
        num_frames=frames_per_clip,
        tubelet_size=tubelet_size,
        uniform_power=uniform_power,
        use_sdpa=use_sdpa,
        use_SiLU=use_SiLU,
        tight_SiLU=tight_SiLU,
    )

    encoder.to(device)
    encoder = load_pretrained(encoder=encoder, pretrained=pretrained, checkpoint_key=checkpoint_key)
    return encoder


def init_opt(
    classifier,
    iterations_per_epoch,
    start_lr,
    ref_lr,
    warmup,
    num_epochs,
    wd=1e-6,
    final_wd=1e-6,
    final_lr=0.0,
    use_bfloat16=False
):
    param_groups = [
        {
            'params': (p for n, p in classifier.named_parameters()
                       if ('bias' not in n) and (len(p.shape) != 1))
        }, {
            'params': (p for n, p in classifier.named_parameters()
                       if ('bias' in n) or (len(p.shape) == 1)),
            'WD_exclude': True,
            'weight_decay': 0
        }
    ]

    logger.info('Using AdamW')
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=int(warmup*iterations_per_epoch),
        start_lr=start_lr,
        ref_lr=ref_lr,
        final_lr=final_lr,
        T_max=int(num_epochs*iterations_per_epoch))
    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=wd,
        final_wd=final_wd,
        T_max=int(num_epochs*iterations_per_epoch))
    scaler = torch.cuda.amp.GradScaler() if use_bfloat16 else None
    return optimizer, scaler, scheduler, wd_scheduler
