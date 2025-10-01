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
    CSVLogger
)

from app_scratch.video_classification_frozen.utils import (
    make_transforms,
    ClipAggregation,
    FrameAggregation
)

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

pp = pprint.PrettyPrinter(indent=4)


def main(args_eval, plotter, resume_preempt=False, debug=False):
    if debug:
        print('\n\n')
        print('========================')
        print('========================')
        print('In DEBUG MODE')
        print('========================')
        print('========================')
        print('\n\n')
        mask_dir_path = os.path.join(args_eval.get('pretrain').get('folder'),args_eval.get('eval_name'),args_eval.get('tag'),'iteration_epoch_clip')
        os.makedirs(mask_dir_path, exist_ok=True)
        logger.info(f"creating  {mask_dir_path} " )
    else:
        print('========================')
        print('========================')
        print('IN CLASSIFICATION FINE TUNING')
        print('========================')
        print('========================')
    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- PRETRAIN
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
    
    blockwise_patch_embed= args_pretrain.get('blockwise_patch_embed', False)  #+
    strategy = args_pretrain.get('strategy','consecutive') # default mri selection is 'consecutive' other is 'skip_1'
    # Optional [for Video model]:
    tubelet_size = args_pretrain.get('tubelet_size', 2)
    pretrain_frames_per_clip = args_pretrain.get('frames_per_clip', 1)
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
    # -- DATA
    args_data = args_eval.get('data')
    train_data_path = [args_data.get('dataset_train')]
    val_data_path = [args_data.get('dataset_val')]
    dataset_type = args_data.get('dataset_type', 'VideoDataset')
    num_classes = args_data.get('num_classes')
    num_workers = args_data.get('num_workers', 1)
    eval_num_segments = args_data.get('num_segments', 1)
    eval_frames_per_clip = args_data.get('frames_per_clip', 16)
    eval_frame_step = args_pretrain.get('frame_step', 4)
    eval_duration = args_pretrain.get('clip_duration', None)
    eval_num_views_per_segment = args_data.get('num_views_per_segment', 1)
    

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
    
    # -- LOGGING    
    args_logging = args_eval.get('logging')
    project_name = args_logging.get('project', 'voxel-jepa-fine-tuning')
    run_name = args_logging.get('run_name', 'voxel-finetune-test')
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
    latest_path = os.path.join(folder, f'{tag}-latest.pth.tar')
    latest_encoder_path = os.path.join(folder, f'{tag}-latest-encoder.pth.tar')
    best_path = os.path.join(folder, f'{tag}-best.pth.tar')
    best_encoder_path = os.path.join(folder, f'{tag}-best-encoder.pth.tar')
    if os.path.exists(latest_encoder_path):
        pretrained_path = latest_encoder_path
    else:
        pretrained_path = os.path.join(pretrain_folder, ckp_fname)

    if rank == 0:
        if plotter == "wandb":
            import wandb
            wandb.init(
                entity= "voxel-jepa" , # "hbgallella",
                project=project_name,
                name=run_name,
                config=args_eval
            )
            from src.utils.logging import WandBCSVLoggerEval
            csv_logger = WandBCSVLoggerEval(csv_path=log_file)
        else:
            csv_logger = CSVLogger(
                log_file,
                ('%d', 'epoch'),
                ('%.5f', 'train_acc'),
                ('%.5f', 'val_acc'),
                ('%.5f', 'loss')
            )
    #######################

    # Initialize model
    ##change is here.

    # -- pretrained encoder (frozen)
    encoder = init_model(
        crop_size=resolution,
        device=device,
        resume_checkpoint=resume_checkpoint,
        pretrained=pretrained_path,
        model_name=model_name,
        patch_size=patch_size,
        tubelet_size=tubelet_size,
        frames_per_clip=pretrain_frames_per_clip,
        uniform_power=uniform_power,
        checkpoint_key=checkpoint_key,
        use_SiLU=use_SiLU,
        tight_SiLU=tight_SiLU,
        use_sdpa=use_sdpa,
        blockwise_patch_embed=blockwise_patch_embed, #+

    )
    import pdb; pdb.set_trace()
    # check for untrainable parameters
    for name, param in encoder.named_parameters():
        if not param.requires_grad:
            print(f"{name} with shape {param.shape}")
            # param.requires_grad = True
        # # import pdb; pdb.set_trace()
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
    # encoder.eval() ## as this is a scratch model, we just allow gradients to pass through
    # for p in encoder.parameters():
    #     p.requires_grad = False

    # -- init classifier
    classifier = AttentiveClassifier(
        embed_dim=encoder.embed_dim,
        num_heads=encoder.num_heads,
        depth=1,
        num_classes=num_classes,
    ).to(device)

    train_loader = make_dataloader(
        dataset_type=dataset_type,
        root_path=train_data_path,
        resolution=resolution,
        frames_per_clip=eval_frames_per_clip,
        frame_step=eval_frame_step,
        eval_duration=eval_duration,
        num_segments=eval_num_segments if attend_across_segments else 1,
        num_views_per_segment=1,
        allow_segment_overlap=True,
        batch_size=batch_size,
        world_size=world_size,
        rank=rank,
        training=True,
        data_aug_dict=data_aug_dict,
        strategy=strategy) #+
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
        data_aug_dict=data_aug_dict,
        strategy=strategy) #+
    ipe = len(train_loader)
    logger.info(f'Dataloader created... iterations per epoch: {ipe}')
    # Re-run the check to confirm
    def check_all_requires_grad(model, model_name):
        all_params = list(model.named_parameters())
        total_params = len(all_params)
        trainable_params = sum(1 for _, p in all_params if p.requires_grad)
        all_trainable = total_params == trainable_params
        print(f"{model_name}: {total_params} total params, {trainable_params} trainable, "
            f"All requires_grad=True: {all_trainable}")
        if not all_trainable:
            non_trainable = [(name, p.shape) for name, p in all_params if not p.requires_grad][:5]
            print(f"Non-trainable params (first 5): {non_trainable}")
        return all_trainable

    encoder_trainable = check_all_requires_grad(encoder, "Encoder")   
    classifier_trainable = check_all_requires_grad(classifier, "classifier") 
    # import pdb; pdb.set_trace()
    # -- optimizer and scheduler
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        encoder=encoder,
        classifier=classifier,
        wd=wd,
        start_lr=start_lr,
        ref_lr=lr,
        final_lr=final_lr,
        iterations_per_epoch=ipe,
        warmup=warmup,
        num_epochs=num_epochs,
        use_bfloat16=use_bfloat16)
    classifier = DistributedDataParallel(classifier, static_graph=True)

    # -- load training checkpoint
    start_epoch = 0
    if resume_checkpoint:
        # Print first classifier param before loading
        for name, param in classifier.named_parameters():
            print("Before loading:", name, param.flatten()[:5])
            break
        classifier, optimizer, scaler, start_epoch = load_checkpoint(
            device=device,
            r_path=latest_path,
            classifier=classifier,
            opt=optimizer,
            scaler=scaler)
        # Print first classifier param after loading
        for name, param in classifier.named_parameters():
            print("After loading:", name, param.flatten()[:5])
            break
        for _ in range(start_epoch*ipe):
            scheduler.step()
            wd_scheduler.step()
    pdb.set_trace()
    def save_encoder_checkpoint(epoch):
        if rank != 0:
            return
        save_dict = {
            'encoder': encoder.state_dict(),
            'opt': optimizer.state_dict(),
            'scaler': None if scaler is None else scaler.state_dict(),
            'epoch': epoch,
            'batch_size': batch_size,
            'world_size': world_size,
            'lr': lr,
        }
        try:
            torch.save(save_dict, latest_encoder_path)
        except Exception as e:
            logger.info(f'Encountered exception when saving checkpoint: {e}')

    def save_checkpoint(epoch):
        save_dict = {
            'classifier': classifier.state_dict(),
            'opt': optimizer.state_dict(),
            'scaler': None if scaler is None else scaler.state_dict(),
            'epoch': epoch,
            'batch_size': batch_size,
            'world_size': world_size,
            'lr': lr
        }
        if rank == 0:
            torch.save(save_dict, latest_path)

    def save_best_encoder_checkpoint(epoch):
        if rank != 0:
            return
        save_dict = {
            'encoder': encoder.state_dict(),
            'opt': optimizer.state_dict(),
            'scaler': None if scaler is None else scaler.state_dict(),
            'epoch': epoch,
            'batch_size': batch_size,
            'world_size': world_size,
            'lr': lr,
        }
        try:
            torch.save(save_dict, best_encoder_path)
        except Exception as e:
            logger.info(f'Encountered exception when saving checkpoint: {e}')

    def save_best_checkpoint(epoch):
        save_dict = {
            'classifier': classifier.state_dict(),
            'opt': optimizer.state_dict(),
            'scaler': None if scaler is None else scaler.state_dict(),
            'epoch': epoch,
            'batch_size': batch_size,
            'world_size': world_size,
            'lr': lr
        }
        if rank == 0:
            torch.save(save_dict, best_path)
    # TRAIN LOOP
    best_acc = float('-inf')
    prev_encoder_weights = [p.clone().detach() for p in encoder.parameters()]
    prev_classifier_weights = [p.clone().detach() for p in classifier.parameters()]
    for epoch in range(start_epoch, num_epochs):
        logger.info('Epoch %d' % (epoch + 1))
        train_acc, loss, grad_norm = run_one_epoch(
            device=device,
            training=True,
            num_temporal_views=eval_num_segments if attend_across_segments else 1,
            attend_across_segments=attend_across_segments,
            num_spatial_views=1,
            encoder=encoder,
            classifier=classifier,
            scaler=scaler,
            optimizer=optimizer,
            scheduler=scheduler,
            wd_scheduler=wd_scheduler,
            data_loader=train_loader,
            use_bfloat16=use_bfloat16,
            debug=debug,
            mask_dir_path=mask_dir_path if debug else None)

        val_acc, _ , _ = run_one_epoch(
            device=device,
            training=False,
            num_temporal_views=eval_num_segments,
            attend_across_segments=attend_across_segments,
            num_spatial_views=eval_num_views_per_segment,
            encoder=encoder,
            classifier=classifier,
            scaler=scaler,
            optimizer=optimizer,
            scheduler=scheduler,
            wd_scheduler=wd_scheduler,
            data_loader=val_loader,
            use_bfloat16=use_bfloat16,
            debug=debug,
            mask_dir_path=mask_dir_path if debug else None)

        logger.info('[%5d] train: %.3f%% test: %.3f%% (loss: %.3f, grad_norm: %.3f)' % (epoch + 1, train_acc, val_acc, loss, grad_norm))
        if rank == 0:
            csv_logger.log(epoch + 1, train_acc, val_acc, loss, grad_norm)
        save_checkpoint(epoch + 1)
        save_encoder_checkpoint(epoch + 1)
        if val_acc > best_acc:
            best_acc = val_acc
            save_best_checkpoint(epoch + 1)
            save_best_encoder_checkpoint(epoch + 1)
        # --- Weight change check ---
        curr_encoder_weights = [p.clone().detach() for p in encoder.parameters()]
        curr_classifier_weights = [p.clone().detach() for p in classifier.parameters()]

        encoder_changed = any([(not torch.equal(b, a)) for b, a in zip(prev_encoder_weights, curr_encoder_weights)])
        classifier_changed = any([(not torch.equal(b, a)) for b, a in zip(prev_classifier_weights, curr_classifier_weights)])

        print(f"Epoch {epoch+1}: Encoder weights changed: {encoder_changed}, Classifier weights changed: {classifier_changed}")

        # Update for next epoch
        prev_encoder_weights = curr_encoder_weights
        prev_classifier_weights = curr_classifier_weights    
    ### New part by Hasitha
    if rank == 0 and args_eval.get("plotter", "csv") == "wandb":
        import wandb
        wandb.finish()
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
    debug=False,
    mask_dir_path=None
):
    encoder.train(mode=training)
    classifier.train(mode=training)
    criterion = torch.nn.CrossEntropyLoss()
    top1_meter = AverageMeter()
    loss_meter = AverageMeter()
    grad_meter = AverageMeter()
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
            
            outputs = encoder(clips, clip_indices)
            # import pdb; pdb.set_trace()
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
        # logits = outputs[0][0] if not attend_across_segments else outputs[0]
        # print(f"Iter {itr}: Classifier logits min={logits.min().item()}, max={logits.max().item()}")
        # import pdb; pdb.set_trace()
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
            loss_meter.update(float(loss), batch_size)
        # print("Input min/max:", clips[0][0].min().item(), clips[0][0].max().item())
        # print("Labels:", labels)
        # print("Logits min/max:", outputs.min().item(), outputs.max().item())
        # print("Loss:", loss.item())
        # import pdb; pdb.set_trace()
        if training:
            if use_bfloat16:
                scaler.scale(loss).backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(classifier.parameters()), 1.0)
                for name, param in list(encoder.named_parameters()) + list(classifier.named_parameters()):
                    if param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            print(f"NaN or Inf in gradient of {name}")
                # print("Loss:", loss.item())
                # print(f"Iter {itr}: Loss={loss.item():.4f}, Grad norm={grad_norm.item():.4f}, "
                #     f"Scaler scale={scaler.get_scale()}")
                # print(f"Iter {itr}: LR={optimizer.param_groups[0]['lr']:.6f}")   
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                for name, param in list(encoder.named_parameters()) + list(classifier.named_parameters()):
                    if param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            print(f"NaN or Inf in gradient of {name}")
                # import pdb; pdb.set_trace()
                torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(classifier.parameters()), 1.0)

                optimizer.step()

                # encoder_weights_after = [p.clone().detach() for p in encoder.parameters()]
                # classifier_weights_after = [p.clone().detach() for p in classifier.parameters()]
                # any_change = any([(not torch.equal(b, a)) for b, a in zip(encoder_weights_before, encoder_weights_after)])
                # print(f"Encoder weights updated: {any_change}")
                # any_classifier_change = any([(not torch.equal(b, a)) for b, a in zip(classifier_weights_before, classifier_weights_after)])
                # print(f"Classifier weights updated: {any_classifier_change}")
                # import pdb; pdb.set_trace()

            grad_meter.update(grad_norm, batch_size)
            optimizer.zero_grad()
        if itr % 20 == 0:
            logger.info('[%5d] %.3f%% (loss: %.3f) [mem: %.2e]'
                        % (itr, top1_meter.avg, loss_meter.avg,
                           torch.cuda.max_memory_allocated() / 1024.**2))
    return top1_meter.avg , loss_meter.avg , grad_meter.avg


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
        random_resize_aspect_ratio=ar_range,
        random_resize_scale=rr_scale,
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
        strategy=strategy,) #+
    return data_loader


def init_model(
    device,
    pretrained,
    resume_checkpoint,
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
    checkpoint_key='target_encoder',
    blockwise_patch_embed=False #+
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
        blockwise_patch_embed=blockwise_patch_embed, #+
    )

    encoder.to(device)
    
    # Load pretrained weights if provided
    if pretrained and os.path.exists(pretrained) and resume_checkpoint:
        try:
            # Check if it's an encoder checkpoint (has 'encoder' key) or pretrained (has checkpoint_key)
            checkpoint = torch.load(pretrained, map_location='cpu')
            if 'encoder' in checkpoint:
                # This is an encoder checkpoint saved by save_encoder_checkpoint
                logger.info(f'Loading encoder checkpoint from {pretrained}')
                encoder_dict = checkpoint['encoder']
                encoder_dict = {k.replace('module.', '').replace('model.', ''): v for k, v in encoder_dict.items()}
                msg = encoder.load_state_dict(encoder_dict, strict=False)
                logger.info(f'Loaded encoder checkpoint with msg: {msg}')
                logger.info(f'Loaded encoder from epoch: {checkpoint.get("epoch", "unknown")}')
            else:
                # This is a pretrained checkpoint
                encoder = load_pretrained(encoder=encoder, pretrained=pretrained, checkpoint_key=checkpoint_key)
            del checkpoint
        except Exception as e:
            logger.warning(f'Could not load pretrained weights from {pretrained}: {e}')
    
    return encoder


def init_opt(
    encoder,
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
    def named_params(modules):
        for module in modules:
            for n, p in module.named_parameters():
                yield f"{module.__class__.__name__}.{n}", p

    param_groups = [
        {
            'params': (p for n, p in named_params([encoder, classifier])
                       if ('bias' not in n) and (len(p.shape) != 1))
        }, {
            'params': (p for n, p in named_params([encoder, classifier])
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

def save_clip_frames_as_grid(tensor,label, filename="clip0.png"):
    import matplotlib.pyplot as plt
    import numpy as np

    # Ensure parent directories exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    """
    tensor: torch.Tensor of shape [3, T, 224, 224]
    Saves a grid of T frames (each as RGB) to filename.
    """

    tensor = tensor.detach().cpu()
    T = tensor.shape[1]
    fig, axs = plt.subplots(1, T, figsize=(T * 2, 2))
    if T == 1:
        axs = [axs]
    for i in range(T):
        # [3, 224, 224] -> [224, 224, 3]
        frame = tensor[:, i, :, :].permute(1, 2, 0).numpy()
        # Normalize to [0, 1] for display if needed
        frame = (frame - frame.min()) / (frame.max() - frame.min() + 1e-5)
        axs[i].imshow(frame)
        axs[i].axis('off')
        axs[i].set_title(f"Frame {i}")
    fig.suptitle(str(label))
    plt.tight_layout()
    plt.savefig(filename,dpi=80)
    plt.close(fig)