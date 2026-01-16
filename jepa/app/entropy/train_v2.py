
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
from einops import rearrange

# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
try:
    # -- WARNING: IF DOING DISTRIBUTED TRAINING ON A NON-SLURM CLUSTER, MAKE
    # --          SURE TO UPDATE THIS TO GET LOCAL-RANK ON NODE, OR ENSURE
    # --          THAT YOUR JOBS ARE LAUNCHED WITH ONLY 1 DEVICE VISIBLE
    # --          TO EACH PROCESS
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['SLURM_LOCALID']
except Exception:
    pass

import copy
import time
import numpy as np

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

from src.datasets.data_manager import init_data
from src.masks.random_tube import MaskCollator as TubeMaskCollator
from src.masks.multiblock3d import MaskCollator as MB3DMaskCollator
from src.masks.half_mask import MaskCollator as HalfMaskCollator
from src.masks.utils import apply_masks
from src.utils.distributed import init_distributed, AllReduce
from src.utils.logging import (
    CSVLogger,
    gpu_timer,
    get_logger,
    grad_logger,
    adamw_logger,
    AverageMeter)
from src.utils.tensors import repeat_interleave_batch

from app.entropy.utils import (
    load_checkpoint,
    init_video_model,
    init_decoder,
    init_opt,
)
from app.entropy.transforms import make_transforms
from src.utils.entropy_loss import calculate_trace_torch


# --
log_timings = True
log_freq = 10
checkpoint_freq = 1
# --

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True


logger = get_logger(__name__)


def main(args, resume_preempt=False,debug=False,save_mask=False):
    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #
    if debug:
        print('\n\n')
        print('========================')
        print('========================')
        print('DEBUG')
        print('========================')
        print('========================')
        print('\n\n')
        mask_dir_path = os.path.join(args.get('logging').get('folder'),'iteration_epoch_clip')
        os.makedirs(mask_dir_path, exist_ok=True)
        logger.info(f"creating  {mask_dir_path} " )
    if save_mask:
        from src.utils.save_mask import save_masks_to_csv
        # Debug phase: Save masks_enc and masks_pred to a CSV file
        mask_dir_path = os.path.join(args.get('logging').get('folder'),'masks_csv')
        os.makedirs(mask_dir_path, exist_ok=True)

        # print(f'SAVING MASKS: masks_csv folder created in : {mask_dir_path}')
    # -- META
    cfgs_meta = args.get('meta')
    load_model = cfgs_meta.get('load_checkpoint') or resume_preempt
    r_file = cfgs_meta.get('read_checkpoint', None)
    seed = cfgs_meta.get('seed', _GLOBAL_SEED)
    save_every_freq = cfgs_meta.get('save_every_freq', -1)
    skip_batches = cfgs_meta.get('skip_batches', -1)
    use_sdpa = cfgs_meta.get('use_sdpa', False)
    which_dtype = cfgs_meta.get('dtype')
    logger.info(f'{which_dtype=}')
    if which_dtype.lower() == 'bfloat16':
        dtype = torch.bfloat16
        mixed_precision = True
    elif which_dtype.lower() == 'float16':
        dtype = torch.float16
        mixed_precision = True
    else:
        dtype = torch.float32
        mixed_precision = False
    # -- MASK
    cfgs_mask = args.get('mask')

    # -- MODEL
    cfgs_model = args.get('model')
    model_name = cfgs_model.get('model_name')
    pred_depth = cfgs_model.get('pred_depth')
    pred_embed_dim = cfgs_model.get('pred_embed_dim')
    uniform_power = cfgs_model.get('uniform_power', True)
    use_mask_tokens = cfgs_model.get('use_mask_tokens', True)
    zero_init_mask_tokens = cfgs_model.get('zero_init_mask_tokens', True)
    # blockwise_patch_embed= cfgs_model.get('blockwise_patch_embed', False)  #+

    # -- DATA
    cfgs_data = args.get('data')
    decoder_only_epochs = cfgs_data.get('decoder_only_epochs', 5)
    atlas_axis_0_path = cfgs_data.get('atlas')[0]
    atlas_axis_1_path = cfgs_data.get('atlas')[1]
    atlas_axis_2_path = cfgs_data.get('atlas')[2]
    cov_loss_weight = cfgs_data.get('cov_loss_weight', 0.1)
    dataset_type = cfgs_data.get('dataset_type', 'videodataset')
    mask_type = cfgs_data.get('mask_type', 'multiblock3d')
    dataset_paths = cfgs_data.get('datasets', [])
    datasets_weights = cfgs_data.get('datasets_weights', None)
    if datasets_weights is not None:
        assert len(datasets_weights) == len(dataset_paths), 'Must have one sampling weight specified for each dataset'
    batch_size = cfgs_data.get('batch_size')
    num_clips = cfgs_data.get('num_clips')
    num_frames = cfgs_data.get('num_frames')
    tubelet_size = cfgs_data.get('tubelet_size')
    sampling_rate = cfgs_data.get('sampling_rate')
    duration = cfgs_data.get('clip_duration', None)
    crop_size = cfgs_data.get('crop_size', 224)
    patch_size = cfgs_data.get('patch_size')
    pin_mem = cfgs_data.get('pin_mem', False)
    num_workers = cfgs_data.get('num_workers', 1)
    filter_short_videos = cfgs_data.get('filter_short_videos', False)
    decode_one_clip = cfgs_data.get('decode_one_clip', True)
    log_resource_util_data = cfgs_data.get('log_resource_utilization', False)
    strategy = cfgs_data.get('strategy','consecutive') # default mri selection is 'consecutive' other is 'skip_1'

    # -- DATA AUGS
    cfgs_data_aug = args.get('data_aug')
    ar_range = cfgs_data_aug.get('random_resize_aspect_ratio', [3/4, 4/3])
    rr_scale = cfgs_data_aug.get('random_resize_scale', [0.3, 1.0])
    motion_shift = cfgs_data_aug.get('motion_shift', False)
    reprob = cfgs_data_aug.get('reprob', 0.)
    use_aa = cfgs_data_aug.get('auto_augment', False)
    tensor_normalize = cfgs_data_aug.get('normalize', ((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))) #use Imagenet if nor provided
    # -- LOSS
    cfgs_loss = args.get('loss')
    loss_exp = cfgs_loss.get('loss_exp')
    reg_coeff = cfgs_loss.get('reg_coeff')
    pixel_loss_weight = cfgs_loss.get('pixel_loss_weight')
    perceptual_loss_weight = cfgs_loss.get('perceptual_loss_weight')

    # -- OPTIMIZATION
    cfgs_opt = args.get('optimization')
    ipe = cfgs_opt.get('ipe', None)
    ipe_scale = cfgs_opt.get('ipe_scale', 1.0)
    clip_grad = cfgs_opt.get('clip_grad', None)
    wd = float(cfgs_opt.get('weight_decay'))
    final_wd = float(cfgs_opt.get('final_weight_decay'))
    num_epochs = cfgs_opt.get('epochs')
    warmup = cfgs_opt.get('warmup')
    start_lr = cfgs_opt.get('start_lr')
    lr = cfgs_opt.get('lr')
    final_lr = cfgs_opt.get('final_lr')
    ema = cfgs_opt.get('ema')
    betas = cfgs_opt.get('betas', (0.9, 0.999))
    eps = cfgs_opt.get('eps', 1.e-8)

    # -- LOGGING
    cfgs_logging = args.get('logging')
    folder = cfgs_logging.get('folder')
    tag = cfgs_logging.get('write_tag')

    # ----------------------------------------------------------------------- #
    # ----------------------------------------------------------------------- #

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    try:
        mp.set_start_method('spawn')
    except Exception:
        pass

    # -- init torch distributed backend
    world_size, rank = init_distributed()
    logger.info(f'Initialized (rank/world-size) {rank}/{world_size}')

    # -- set device
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)

    # -- log/checkpointing paths
    log_file = os.path.join(folder, f'{tag}_r{rank}.csv')
    latest_file = f'{tag}-latest.pth.tar'
    latest_path = os.path.join(folder, latest_file)
    load_path = None
    if load_model:
        load_path = os.path.join(folder, r_file) if r_file is not None else latest_path
        if not os.path.exists(load_path):
            load_path = None
            load_model = False

    # -- make csv_logger
    # csv_logger = CSVLogger(
    #     log_file,
    #     ('%d', 'epoch'),
    #     ('%d', 'itr'),
    #     ('%.5f', 'loss'),
    #     ('%.5f', 'loss-jepa'),
    #     ('%.5f', 'reg-loss'),
    #     ('%.5f', 'enc-grad-norm'),
    #     ('%.5f', 'pred-grad-norm'),
    #     ('%d', 'gpu-time(ms)'),
    #     ('%d', 'wall-time(ms)'),
    # )
    ######## New by Hasitha ####
    if args.get("plotter", "csv") == "wandb" and rank == 0:
        from src.utils.logging import WandBCSVLogger
        logger_impl = WandBCSVLogger(csv_path=log_file)
    else:
        logger_impl = CSVLogger(
            log_file,
            ('%d', 'epoch'),
            ('%d', 'itr'),
            ('%.5f', 'loss'),
            ('%.5f', 'loss-jepa'),
            ('%.5f', 'reg-loss'),
            ('%.5f', 'loss-pixel'),  # Add loss_pixel here
            ('%.5f', 'enc-grad-norm'),
            ('%.5f', 'pred-grad-norm'),
            ('%.5f', 'cov-loss'),  # Add cov_loss here
            ('%d', 'gpu-time(ms)'),
            ('%d', 'wall-time(ms)'),
        )

    ############################

    # -- init model
    encoder, predictor = init_video_model(
        uniform_power=uniform_power,
        use_mask_tokens=use_mask_tokens,
        num_mask_tokens=len(cfgs_mask),
        zero_init_mask_tokens=zero_init_mask_tokens,
        device=device,
        patch_size=patch_size,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        model_name=model_name,
        crop_size=crop_size,
        pred_depth=pred_depth,
        pred_embed_dim=pred_embed_dim,
        use_sdpa=use_sdpa,
        # blockwise_patch_embed= blockwise_patch_embed, #+
    )
    target_encoder = copy.deepcopy(encoder)

    # -- init decoder (NEW)
    decoder = None
    decoder_enabled = args.get('decoder', {}).get('enabled', False)
    if decoder_enabled:
        decoder_config = args.get('decoder', {})
        decoder_loss_type = decoder_config.get('loss_type')
        decoder = init_decoder(
            device=device,
            patch_size=patch_size,
            num_frames=num_frames,
            tubelet_size=tubelet_size,
            model_name=model_name,
            crop_size=crop_size,
            decoder_type=decoder_config.get('decoder_type', 'videodecoder'),
            decoder_depth=decoder_config.get('decoder_depth', 4),
            decoder_heads=decoder_config.get('decoder_heads', 12),
            decoder_dim=decoder_config.get('decoder_dim', 512),
            in_chans=3,
        )
        logger.info('Decoder initialized successfully')

    # -- make data transforms
    if mask_type == 'multiblock3d':
        logger.info('Initializing basic multi-block mask')
        mask_collator = MB3DMaskCollator(
            crop_size=crop_size,
            num_frames=num_frames,
            patch_size=patch_size,
            tubelet_size=tubelet_size,
            cfgs_mask=cfgs_mask,
            debug=debug)
    elif mask_type == 'half':
        logger.info('Initializing half mask')
        mask_collator = HalfMaskCollator(
            crop_size=crop_size,
            num_frames=num_frames,
            patch_size=patch_size,
            tubelet_size=tubelet_size,
            cfgs_mask=cfgs_mask,
            debug=debug)
    else:
        logger.info('Initializing random tube mask')
        mask_collator = TubeMaskCollator(
            crop_size=crop_size,
            num_frames=num_frames,
            patch_size=patch_size,
            tubelet_size=tubelet_size,
            cfgs_mask=cfgs_mask)
    transform = make_transforms(
        random_horizontal_flip=True,
        random_resize_aspect_ratio=ar_range,
        random_resize_scale=rr_scale,
        reprob=reprob,
        auto_augment=use_aa,
        motion_shift=motion_shift,
        crop_size=crop_size,
        normalize=tensor_normalize)

    # -- init data-loaders/samplers
    (unsupervised_loader,
     unsupervised_sampler) = init_data(
        data=dataset_type,
        root_path=dataset_paths,
        batch_size=batch_size,
        training=True,
        clip_len=num_frames,
        frame_sample_rate=sampling_rate,
        filter_short_videos=filter_short_videos,
        decode_one_clip=decode_one_clip,
        duration=duration,
        num_clips=num_clips,
        transform=transform,
        datasets_weights=datasets_weights,
        collator=mask_collator,
        num_workers=num_workers,
        world_size=world_size,
        pin_mem=pin_mem,
        rank=rank,
        log_dir=folder if log_resource_util_data else None,
        strategy=strategy,)
    try:
        _dlen = len(unsupervised_loader)
    except Exception:  # Different interface for webdataset
        _dlen = unsupervised_loader.num_batches
    if ipe is None:
        ipe = _dlen
    logger.info(f'iterations per epoch/dataest length: {ipe}/{_dlen}')

    # -- init optimizer and scheduler (JEPA only)
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        encoder=encoder,
        predictor=predictor,
        decoder=None,  # JEPA optimizer (no decoder)
        wd=wd,
        final_wd=final_wd,
        start_lr=start_lr,
        ref_lr=lr,
        final_lr=final_lr,
        iterations_per_epoch=ipe,
        warmup=warmup,
        num_epochs=num_epochs,
        ipe_scale=ipe_scale,
        mixed_precision=mixed_precision,
        betas=betas,
        eps=eps)

    # NEW: Separate decoder optimizer (if decoder enabled)
    decoder_optimizer = None
    decoder_scaler = None
    decoder_scheduler = None
    decoder_wd_scheduler = None

    if decoder is not None:
        decoder_config = args.get('decoder', {})
        decoder_lr = decoder_config.get('lr', lr * 0.1)  # Lower LR for decoder
        decoder_wd = decoder_config.get('weight_decay', wd)
        decoder_warmup = decoder_config.get('warmup', warmup)

        decoder_optimizer, decoder_scaler, decoder_scheduler, decoder_wd_scheduler = init_opt(
            encoder=None,
            predictor=None,
            decoder=decoder,  # Decoder-only optimizer
            wd=decoder_wd,
            final_wd=final_wd,
            start_lr=start_lr,
            ref_lr=decoder_lr,
            final_lr=final_lr,
            iterations_per_epoch=ipe,
            warmup=decoder_warmup,
            num_epochs=num_epochs,
            ipe_scale=ipe_scale,
            mixed_precision=mixed_precision,
            betas=betas,
            eps=eps)

        logger.info(f'Decoder optimizer initialized with LR: {decoder_lr}, WD: {decoder_wd}')
    encoder = DistributedDataParallel(encoder, static_graph=True)
    predictor = DistributedDataParallel(predictor, static_graph=True)
    target_encoder = DistributedDataParallel(target_encoder)
    for p in target_encoder.parameters():
        p.requires_grad = False

    # NEW: Add decoder to DDP if enabled
    if decoder is not None:
        decoder = DistributedDataParallel(decoder, static_graph=True)

    # -- momentum schedule
    momentum_scheduler = (ema[0] + i*(ema[1]-ema[0])/(ipe*num_epochs*ipe_scale)
                          for i in range(int(ipe*num_epochs*ipe_scale)+1))

    start_epoch = 0
    # -- load training checkpoint
    if load_model or os.path.exists(latest_path):
        (
            encoder,
            predictor,
            target_encoder,
            optimizer,
            scaler,
            start_epoch,
        ) = load_checkpoint(
            r_path=load_path,
            encoder=encoder,
            predictor=predictor,
            target_encoder=target_encoder,
            opt=optimizer,
            scaler=scaler,
            decoder=decoder,
            decoder_opt=decoder_optimizer,
            decoder_scaler=decoder_scaler)  # NEW: Include decoder optimizers
        for _ in range(start_epoch * ipe):
            scheduler.step()
            wd_scheduler.step()
            next(momentum_scheduler)
            mask_collator.step()

    def save_checkpoint(epoch, path):
        if rank != 0:
            return
        save_dict = {
            'encoder': encoder.state_dict(),
            'predictor': predictor.state_dict(),
            'opt': optimizer.state_dict(),
            'scaler': None if scaler is None else scaler.state_dict(),
            'target_encoder': target_encoder.state_dict(),
            'decoder': decoder.state_dict() if decoder is not None else None,
            'decoder_opt': decoder_optimizer.state_dict() if decoder_optimizer is not None else None,  # NEW
            'decoder_scaler': None if decoder_scaler is None else decoder_scaler.state_dict(),  # NEW
            'epoch': epoch,
            'loss': loss_meter.avg,
            'batch_size': batch_size,
            'world_size': world_size,
            'lr': lr,
        }
        try:
            torch.save(save_dict, path)
        except Exception as e:
            logger.info(f'Encountered exception when saving checkpoint: {e}')

    logger.info('Initializing loader...')
    loader = iter(unsupervised_loader)

    if skip_batches > 0:
        logger.info(f'Skip {skip_batches} batches')
        unsupervised_sampler.set_epoch(start_epoch)
        for itr in range(skip_batches):
            if itr % 10 == 0:
                logger.info(f'Skip {itr}/{skip_batches} batches')
            try:
                udata = next(loader)
            except Exception:
                loader = iter(unsupervised_loader)
                udata = next(loader)
    def load_atlas_torch(atlas_path):
        import numpy as np
        from skimage.transform import resize
        import ants
        atlas_nii = ants.image_read(atlas_path)
        atlas_data = atlas_nii.numpy()
        atlas_resized = np.array([resize(slice_c, (224, 224), order=0, preserve_range=True, anti_aliasing=False) for slice_c in atlas_data])
        atlas_tensor = torch.tensor(atlas_resized, dtype=torch.bfloat16, device='cuda')
        return atlas_tensor
    # preload all atlas
    atlas_resized_axis_0 = load_atlas_torch(atlas_axis_0_path).to(device)
    atlas_resized_axis_1 = load_atlas_torch(atlas_axis_1_path).to(device)
    atlas_resized_axis_2 = load_atlas_torch(atlas_axis_2_path).to(device)
    atlas_list = [atlas_resized_axis_0, atlas_resized_axis_1, atlas_resized_axis_2]
    # -- TRAINING LOOP
    for epoch in range(start_epoch, num_epochs):

        logger.info('Epoch %d' % (epoch + 1))
        # if debug:
        #     mask_dir_path = os.path.join(args.get('logging').get('folder'),'masks_csv')
        #     epoch_dir_path = os.path.join(mask_dir_path, f"epoch_{epoch+1}")
        #     os.makedirs(epoch_dir_path, exist_ok=True)
        if save_mask:
            epoch_dir_path = os.path.join(mask_dir_path, f"epoch_{epoch+1}")
            os.makedirs(epoch_dir_path, exist_ok=True)
            clip_index_list = []


        # if debug:

        #     for a1, a2, a3 in unsupervised_loader:
        #         print(epoch, a1[3])
        #         print([a.shape for a in a1[0]],a1[0])
        # -- update distributed-data-loader epoch
        unsupervised_sampler.set_epoch(epoch)
        # if debug:
        #     indices = list(unsupervised_sampler)
        #     logger.info(f"DEBUG: Sampler indices for epoch {epoch}: {indices}")

        loss_meter = AverageMeter()
        cov_loss_meter = AverageMeter()  # NEW: Add a meter for cov_loss
        input_var_meter = AverageMeter()
        input_var_min_meter = AverageMeter()
        jepa_loss_meter = AverageMeter()
        reg_loss_meter = AverageMeter()
        pixel_loss_meter = AverageMeter()  # NEW: Add pixel loss meter
        mask_meters = [AverageMeter() for _ in range(len(cfgs_mask))]
        gpu_time_meter = AverageMeter()
        wall_time_meter = AverageMeter()


        for itr in range(ipe):
            try:
                udata, masks_enc, masks_pred = next(loader)
            except Exception:
                logger.info('Exhausted data loaders. Refreshing...')
                loader = iter(unsupervised_loader)
                udata, masks_enc, masks_pred = next(loader)
            assert len(masks_enc) == len(masks_pred), \
                'Currently require num encoder masks = num predictor masks'
            
            # Stack tensors into a single tensor
            slice_index_list = torch.stack(udata[2][0])  # Shape: [24, 24]
            axis_index_list = udata[5].tolist()

            # Transpose the tensor to group by index
            transposed = slice_index_list.t()  # Shape: [24, 24]

            # Split the transposed tensor into individual tensors if needed
            slice_index_list = [transposed[i] for i in range(transposed.size(0))]
            def load_clips():
                # -- unsupervised video clips
                # Put each clip on the GPU and concatenate along batch
                # dimension
                # import pdb; pdb.set_trace()
                clips = torch.cat([u.to(device, non_blocking=True) for u in udata[0]], dim=0)
                # logger.info(f"clips: {clips.size()}")
                # Put each mask-enc/mask-pred pair on the GPU and reuse the
                # same mask pair for each clip
                _masks_enc, _masks_pred = [], []
                for _me, _mp in zip(masks_enc, masks_pred):
                    _me = _me.to(device, non_blocking=True)
                    _mp = _mp.to(device, non_blocking=True)
                    _me = repeat_interleave_batch(_me, batch_size, repeat=num_clips)
                    _mp = repeat_interleave_batch(_mp, batch_size, repeat=num_clips)
                    _masks_enc.append(_me)
                    _masks_pred.append(_mp)

                return (clips, _masks_enc, _masks_pred)

            clips, masks_enc, masks_pred = load_clips()

            if save_mask:
                epoch_dir_path = os.path.join(mask_dir_path, f"epoch_{epoch+1}")
                os.makedirs(epoch_dir_path, exist_ok=True)
                save_masks_to_csv(masks_enc, masks_pred, filename=os.path.join(epoch_dir_path,f"masks_itr_{itr}.csv"),logger=logger)
                clip_index_list.append([itr,udata[4].tolist()])
            itr_start_time = time.time()

            for _i, m in enumerate(mask_meters):
                m.update(masks_enc[_i][0].size(-1))
                # if debug:
                #     print(f"DEBUG mask_meters[{_i}]: {masks_enc[_i][0].size(-1)}")
            def train_step():
                
                _new_lr = scheduler.step()
                _new_wd = wd_scheduler.step()

                # NEW: Decoder scheduler steps (if decoder enabled)
                _new_dec_lr = None
                _new_dec_wd = None
                if decoder_scheduler is not None:
                    _new_dec_lr = decoder_scheduler.step()
                if decoder_wd_scheduler is not None:
                    _new_dec_wd = decoder_wd_scheduler.step()
                # --
                # Number of epochs to train the decoder only
                    # Initialize cov_loss with a default value

                def forward_target(c):
                    """
                    Returns list of tensors of shape [B, N, D], one for each
                    mask-pred.
                    """
                    with torch.no_grad():
                        h = target_encoder(c)
                        h = F.layer_norm(h, (h.size(-1),))  # normalize over feature-dim  [B, N, D]
                        # -- create targets (masked regions of h)
                        h = apply_masks(h, masks_pred, concat=False)

                        return h

                def forward_context(c, h):
                    """
                    Returns list of tensors of shape [B, N, D], one for each
                    mask-pred.
                    """
                    z = encoder(c, masks_enc)
                    z = predictor(z, h, masks_enc, masks_pred)
                    return z

                def loss_fn(z, h):
                    loss = 0.
                    # Compute loss and accumulate for each mask-enc/mask-pred pair
                    for zi, hi in zip(z, h):
                        loss += torch.mean(torch.abs(zi - hi)**loss_exp) / loss_exp
                    # print(f"total loss is {loss}")
                    loss /= len(masks_pred)
                    return loss

                def reg_fn(z):
                    return sum([torch.sqrt(zi.var(dim=1) + 0.0001) for zi in z]) / len(z)


                def splice(clips, reconstructed_video, masks_enc, patch_size, tubelet_size):
                    """
                    Splice the original clips with the reconstructed video using the mask indices,

                    Args:
                        clips: Original video clips [B, C, T, H, W]
                        reconstructed_video: Reconstructed video [B, C, T, H, W]
                        masks_enc: List of [B, N] tensors with mask indices
                        patch_size: Spatial patch size (e.g., 16)
                        tubelet_size: Temporal tubelet size (e.g., 2)

                    Returns:
                        spliced_video: Spliced video [B, C, T, H, W]
                    """
                    B, C, T, H, W = clips.shape

                    # Step 1: Patchify the clips and reconstructed video
                    patches_clips = rearrange(
                        clips,
                        "b c (t p1) (h p2) (w p3) -> b (t h w) (p1 p2 p3 c)",
                        p1=tubelet_size,
                        p2=patch_size,
                        p3=patch_size,
                    )  # Shape: [B, N, patch_volume]

                    patches_recon = rearrange(
                        reconstructed_video,
                        "b c (t p1) (h p2) (w p3) -> b (t h w) (p1 p2 p3 c)",
                        p1=tubelet_size,
                        p2=patch_size,
                        p3=patch_size,
                    )  # Shape: [B, N, patch_volume]

                    # Step 2: Create the spliced patches
                    spliced_patches = patches_recon.clone()  # Start with reconstructed patches
                    batch_idx = torch.arange(B, device=clips.device)[:, None]  # Batch indices
                    if len(masks_enc) > 0:
                        mask = masks_enc[0]  # Use the first mask for splicing
                        # Ensure both tensors have the same dtype
                        patches_clips = patches_clips.to(spliced_patches.dtype)
                        spliced_patches[batch_idx, mask] = patches_clips[batch_idx, mask]  # Replace masked regions

                    # Step 3: Reconstruct the spliced video
                    spliced_video = rearrange(
                        spliced_patches,
                        "b (t h w) (p1 p2 p3 c) -> b c (t p1) (h p2) (w p3)",
                        t=T // tubelet_size,
                        h=H // patch_size,
                        w=W // patch_size,
                        p1=tubelet_size,
                        p2=patch_size,
                        p3=patch_size,
                        c=C,
                    )  # Shape: [B, C, T, H, W]

                    return spliced_video

                cov_loss = 0.0
                # Step 2. Decoder Forward Pass (Pixel Reconstruction) - DECOUPLED
                loss_pixel = 0.
                

                if decoder is not None:
                    with torch.cuda.amp.autocast(dtype=dtype, enabled=mixed_precision):
                        # Decoder training is independent of JEPA

                        vis_save_path = os.path.join(folder, f"{tag}vis/recon_epoch{epoch+1}.png")
                       
                        # # Get full video features (target)                      
                        h = forward_target(clips)
                        z = forward_context(clips, h)
                        
                        # # this is an initial implementation that only takes the first mask prediction and ignores the rest of the multi maskings
                        full_features_pred = F.layer_norm(z[0][1], (z[0][1].size(-1),))
                        full_features = full_features_pred
                        # this keeps computation graphs up to full_features.

                        # DETACH: Prevent any gradient flow into encoder
                        full_features_detached = full_features.detach()

                        # Decode directly
                        reconstructed_video = decoder(full_features_detached)  # [B, 3, T, H, W]
                        loss_pixel = F.mse_loss(reconstructed_video, clips)
 
                        # Optional: perceptual loss
                        perceptual_loss = 0.0
                        if perceptual_loss_weight > 0:
                            perceptual_loss = F.l1_loss(
                                F.avg_pool3d(reconstructed_video, 2),
                                F.avg_pool3d(clips, 2)
                            )
                        loss_pixel = pixel_loss_weight * loss_pixel + perceptual_loss_weight * perceptual_loss

                        # reconstructed_video_pred = decoder(full_features_pred)
                        # reconstructed_video = recon_video
                        # visualize at epoch end


                if epoch >= decoder_only_epochs:
                    
                    # JEPA Forward Pass (Feature Prediction)
                    loss_jepa, loss_reg = 0., 0.
                    with torch.cuda.amp.autocast(dtype=dtype, enabled=mixed_precision):
                        z = [pred[0] for pred in z] # nested list, [pred, full_pred]
                        loss_jepa = loss_fn(z, h)  # jepa prediction loss
                        pstd_z = reg_fn(z)  # predictor variance across patches
                        loss_reg += torch.mean(F.relu(1.-pstd_z))
                        reconstructed_vid = decoder(full_features_pred) # take from previous computation graph.
                        spliced_video = splice(clips, reconstructed_vid, masks_enc, patch_size, tubelet_size)

                        with torch.no_grad():
                            orig_loss= calculate_trace_torch(clips, slice_index_list, axis_index_list, device=clips.device, atlas_list=atlas_list)
                        
                        recon_loss= calculate_trace_torch(spliced_video, slice_index_list, axis_index_list, device=reconstructed_vid.device, atlas_list=atlas_list)
                        cov_loss = nn.L1Loss()(orig_loss, recon_loss) * cov_loss_weight
                        
                        jepa_loss = loss_jepa + reg_coeff * loss_reg + cov_loss


                    if (itr == 0 and rank == 0):
                        visualize_reconstruction_video(
                            clips=clips,
                            reconstructed_video=reconstructed_vid,
                            masks_enc=masks_pred[0],
                            patch_size=patch_size,
                            tubelet_size=tubelet_size,
                            in_chans=3,
                            save_path=vis_save_path,
                            logger=logger,
                            slice_index_list=slice_index_list
                        )
                        
                else:
                    loss_jepa, loss_reg = torch.tensor(0.0, device=clips.device), torch.tensor(0.0, device=clips.device)  

                if epoch >= decoder_only_epochs:
                    pass
                else:
                    jepa_loss = loss_jepa + reg_coeff * loss_reg
                
                if epoch >= decoder_only_epochs:
                    # Step 3. JEPA Backward & Step (UNCHANGED)
                    _enc_norm, _pred_norm = 0., 0.
                    if mixed_precision:
                        
                        scaler.scale(jepa_loss).backward()
                        scaler.unscale_(optimizer)
                    else:
                        jepa_loss.backward()

                    if (epoch > warmup) and (clip_grad is not None):
                        _enc_norm = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip_grad)
                        _pred_norm = torch.nn.utils.clip_grad_norm_(predictor.parameters(), clip_grad)
                    if mixed_precision:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad()
                else:
                    _enc_norm, _pred_norm = 0., 0.
                    # Zero gradients since no JEPA update
                    optimizer.zero_grad()
                # Combined loss for logging
                loss = jepa_loss #jepa_loss + loss_pixel

                # Step 4. Decoder Backward & Step (DECOUPLED)
                _dec_norm = 0.
                if decoder is not None and loss_pixel > 0:
                    if mixed_precision:
                        decoder_scaler.scale(loss_pixel).backward()
                        decoder_scaler.unscale_(decoder_optimizer)
                    else:
                        loss_pixel.backward()
                    if (epoch > warmup) and (clip_grad is not None):
                        _dec_norm = torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip_grad)
                    if mixed_precision:
                        decoder_scaler.step(decoder_optimizer)
                        decoder_scaler.update()
                    else:
                        decoder_optimizer.step()
                    decoder_optimizer.zero_grad()


                # Gradient logging
                grad_stats = grad_logger(encoder.named_parameters())
                grad_stats.global_norm = float(_enc_norm)
                grad_stats_pred = grad_logger(predictor.named_parameters())
                grad_stats_pred.global_norm = float(_pred_norm)
                optim_stats = adamw_logger(optimizer)

                # Decoder gradient logging
                grad_stats_dec = None
                optim_stats_dec = None
                if decoder is not None:
                    grad_stats_dec = grad_logger(decoder.named_parameters())
                    grad_stats_dec.global_norm = float(_dec_norm)
                    optim_stats_dec = adamw_logger(decoder_optimizer)

                # Step 3. momentum update of target encoder
                m = next(momentum_scheduler)
                with torch.no_grad():
                    for param_q, param_k in zip(encoder.parameters(), target_encoder.parameters()):
                        param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)

                return (
                    float(loss),
                    float(loss_jepa),
                    float(loss_reg),
                    float(loss_pixel),  # NEW: Include pixel loss
                    _new_lr,
                    _new_wd,
                    _new_dec_lr,   # ADD THIS
                    _new_dec_wd,   # ADD THIS
                    grad_stats,
                    grad_stats_pred,
                    optim_stats,
                    float(cov_loss),  
                )
            # (loss, loss_jepa, loss_reg, loss_pixel, _new_lr, _new_wd, grad_stats, grad_stats_pred, optim_stats,), gpu_etime_ms = gpu_timer(train_step)
            (loss, loss_jepa, loss_reg, loss_pixel, _new_lr, _new_wd, _new_dec_lr, _new_dec_wd,
             grad_stats, grad_stats_pred, optim_stats, cov_loss), gpu_etime_ms = gpu_timer(train_step)
            iter_elapsed_time_ms = (time.time() - itr_start_time) * 1000.
            loss_meter.update(loss)
            input_var = float(AllReduce.apply(clips.view(clips.shape[0], -1).var(dim=1).mean(dim=0)))
            input_var_min = float(AllReduce.apply(torch.min(clips.view(clips.shape[0], -1).var(dim=1))))
            input_var_meter.update(input_var)
            input_var_min_meter.update(input_var_min)
            jepa_loss_meter.update(loss_jepa)
            reg_loss_meter.update(loss_reg)
            pixel_loss_meter.update(loss_pixel)  # NEW: Update pixel loss meter
            cov_loss_meter.update(cov_loss)  # NEW: Add a meter for cov_loss
            gpu_time_meter.update(gpu_etime_ms)
            wall_time_meter.update(iter_elapsed_time_ms)

            # -- Logging
            def log_stats():
                if rank == 0:
                    logger_impl.log(
                        epoch + 1,
                        itr,
                        loss,
                        loss_jepa,
                        loss_reg,
                        loss_pixel,
                        grad_stats.global_norm,
                        grad_stats_pred.global_norm,
                        cov_loss,  # Log cov_loss
                        gpu_etime_ms,
                        iter_elapsed_time_ms,

                    )

                    if (itr % log_freq == 0) or np.isnan(loss) or np.isinf(loss):
                        # Format decoder LR for logging
                        dec_lr_str = f"dec_lr: {_new_dec_lr:.2e}" if _new_dec_lr is not None else "dec_lr: N/A"

                        logger.info(
                            '[%d, %5d] loss: %.3f | JEPA: p%.3f r%.3f | DECODER: px%.3f | '
                            'COV: %.3f | input_var: %.3f %.3f | '
                            'masks: %s '
                            '[wd: %.2e] [lr: %.2e] [%s] '
                            '[mem: %.2e] '
                            '[gpu: %.1f ms]'
                            '[wall: %.1f ms]'
                            % (epoch + 1, itr,
                            loss_meter.avg,
                            jepa_loss_meter.avg,
                            reg_loss_meter.avg,
                            pixel_loss_meter.avg,  # NEW: Include pixel loss
                            cov_loss,  # Log cov_loss
                            input_var_meter.avg,
                            input_var_min_meter.avg,
                            '[' + ', '.join(['%.1f' % m.avg for m in mask_meters]) + ']',
                            _new_wd,
                            _new_lr,
                            dec_lr_str,
                            torch.cuda.max_memory_allocated() / 1024.0**2,
                            gpu_time_meter.avg,
                            wall_time_meter.avg))

                        if optim_stats is not None:
                            logger.info(
                                '[%d, %5d] first moment: %.2e [%.2e %.2e] second moment: %.2e [%.2e %.2e]'
                                % (epoch + 1, itr,
                                   optim_stats.get('exp_avg').avg,
                                   optim_stats.get('exp_avg').min,
                                   optim_stats.get('exp_avg').max,
                                   optim_stats.get('exp_avg_sq').avg,
                                   optim_stats.get('exp_avg_sq').min,
                                   optim_stats.get('exp_avg_sq').max))

                        if grad_stats is not None:
                            logger.info(
                                '[%d, %5d] enc_grad_stats: f/l[%.2e %.2e] mn/mx(%.2e, %.2e) %.2e'
                                % (epoch + 1, itr,
                                   grad_stats.first_layer,
                                   grad_stats.last_layer,
                                   grad_stats.min,
                                   grad_stats.max,
                                   grad_stats.global_norm))

                        if grad_stats_pred is not None:
                            logger.info(
                                '[%d, %5d] pred_grad_stats: f/l[%.2e %.2e] mn/mx(%.2e, %.2e) %.2e'
                                % (epoch + 1, itr,
                                   grad_stats_pred.first_layer,
                                   grad_stats_pred.last_layer,
                                   grad_stats_pred.min,
                                   grad_stats_pred.max,
                                   grad_stats_pred.global_norm))

            log_stats()
            # assert not np.isnan(loss), 'loss is nan'

        if save_mask:
            with open(os.path.join(epoch_dir_path,"clip_index_list.csv"), "w") as file:
                file.writelines("itr\t clip_index_list\n")
                file.writelines(["\t".join(map(str, row)) + "\n" for row in clip_index_list])
                # -- Save Checkpoint
        logger.info('avg. loss %.3f' % loss_meter.avg)
        if not debug: # since debugging dont save
            # -- Save Last
            if epoch % checkpoint_freq == 0 or epoch == (num_epochs - 1):
                save_checkpoint(epoch + 1, latest_path)
                if save_every_freq > 0 and epoch % save_every_freq == 0:
                    save_every_file = f'{tag}-e{epoch}.pth.tar'
                    save_every_path = os.path.join(folder, save_every_file)
                    save_checkpoint(epoch + 1, save_every_path)

    ##### New by Hasitha ####
    if args.get("plotter", "csv") == "wandb" and rank == 0:
        import wandb
        wandb.finish()


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


def unnormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Unnormalize a tensor using ImageNet mean and std.
    Args:
        tensor: Tensor of shape [C, H, W] or [B, C, H, W]
        mean: Channel-wise mean
        std: Channel-wise std
    Returns:
        Unnormalized tensor clamped to [0, 1]
    """
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor.clamp(0, 1)

def visualize_reconstruction_video(
        clips,
        reconstructed_video,
        masks_enc,
        patch_size,
        tubelet_size,
        in_chans,
        save_path,
        logger=None,
        export_mp4=False,
        slice_index_list=False
):
    """
    Visualize V-JEPA video reconstruction for VideoDecoder output with shape safety.
    Args:
        clips: Ground truth video [B, C, T, H, W]
        reconstructed_video: Reconstructed video [B, C, T, H, W]
        masks_enc: List of [B, N] tensors with patch indices
        patch_size: Spatial patch size
        tubelet_size: Temporal tubelet size
        in_chans: Number of input channels
        save_path: Path to save the visualization
        logger: Optional logger for messages
        export_mp4: Boolean to export an MP4 video
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        import os
        import torch.nn.functional as F
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        device = clips.device
        B, C, T, H, W = clips.shape

        # Detach, move to CPU, and convert to Float32
        clips = clips.detach().cpu().to(dtype=torch.float32)
        recon = reconstructed_video.detach().cpu().to(dtype=torch.float32)
        slice_index_list = slice_index_list = [tensor.detach().cpu() for tensor in slice_index_list]
        # Stack tensors into a single tensor
        stacked_tensor = torch.stack(slice_index_list)  # Shape: [num_tensors, tensor_length]

        # Save the tensors as .npy files
        npy_save_path = save_path.replace(".png", ".npy")
        np.save(npy_save_path, {"clips": clips.numpy(),"slice_index_list": stacked_tensor.numpy(),"reconstructed_video": recon.numpy()})
        if logger:
            logger.info(f"Saved tensors to: {npy_save_path}")
        else:
            print(f"[INFO] Saved tensors to: {npy_save_path}")

        # Compute patch grid dimensions
        num_patches_t = T // tubelet_size  # e.g., 16 // 2 = 8
        num_patches_h = H // patch_size    # e.g., 224 // 16 = 14
        num_patches_w = W // patch_size    # e.g., 224 // 16 = 14
        num_patches = num_patches_t * num_patches_h * num_patches_w  # e.g., 8 * 14 * 14 = 1568

        # Build union mask (1D) from all mask blocks
        mask_union = torch.zeros((B, num_patches), dtype=torch.bool)
        batch_range = torch.arange(B)[:, None]
        for mask_block in masks_enc:
            mask_union[batch_range, mask_block] = True
        mask_union = mask_union.cpu()

        # Select first sample for visualization
        b = 0
        orig = clips[b]        # [C, T, H, W]
        recon_b = recon[b]     # [C, T, H, W]
        mask_b = mask_union[b] # [num_patches]

        # Pick frames to visualize (up to 16 frames)
        frame_indices = np.linspace(0, T - 1, min(T, 16), dtype=int)
        fig, axs = plt.subplots(3, len(frame_indices), figsize=(len(frame_indices) * 2.5, 7))

        for i, t_idx in enumerate(frame_indices):
            # Extract frames and validate shapes
            orig_frame = orig[:, t_idx, :, :].squeeze()
            recon_frame = recon_b[:, t_idx, :, :].squeeze()
            assert orig_frame.shape == (C, H, W) and recon_frame.shape == (C, H, W), \
                f"Shape mismatch: got {orig_frame.shape}, {recon_frame.shape}, expected ({C}, {H}, {W})"

            # Normalize frames
            orig_frame = unnormalize(orig_frame)
            recon_frame = unnormalize(recon_frame)

            # Build spatial mask for this time step
            t_patch_start = (t_idx // tubelet_size) * (num_patches_h * num_patches_w)
            mask_frame = mask_b[t_patch_start:t_patch_start + (num_patches_h * num_patches_w)].float()
            mask_frame = mask_frame.view(num_patches_h, num_patches_w)
            mask_frame = (
                mask_frame.repeat_interleave(patch_size, dim=0)
                .repeat_interleave(patch_size, dim=1)
                .numpy()
            )

            # Build overlay with purple for masked regions
            overlay = orig_frame.clone()  # [C, H, W]
            mask_expanded = torch.tensor(mask_frame)[None, ...].float()  # [1, H, W]
            # Expand mask to [C, H, W] and apply purple [1, 0, 1] per channel
            mask_expanded_3d = mask_expanded.repeat(C, 1, 1)  # [C, H, W]
            overlay = overlay * (1 - mask_expanded_3d) + mask_expanded_3d * torch.tensor([1, 0, 1]).view(C, 1, 1)

            # Visualize
            axs[0, i].imshow(orig_frame.permute(1, 2, 0))
            axs[0, i].set_title(f"Orig F{t_idx}")
            axs[0, i].axis("off")

            axs[1, i].imshow(overlay.permute(1, 2, 0))
            axs[1, i].set_title("Mask Overlay")
            axs[1, i].axis("off")

            axs[2, i].imshow(recon_frame.permute(1, 2, 0))
            axs[2, i].set_title("Recon")
            axs[2, i].axis("off")

        plt.tight_layout()
        plt.savefig(save_path, dpi=120)
        plt.close(fig)

        if logger:
            logger.info(f"Saved reconstruction visualization: {save_path}")
        else:
            print(f"[INFO] Saved reconstruction visualization: {save_path}")

        # Optional MP4 export
        if export_mp4:
            import imageio
            out_path = save_path.replace(".png", ".mp4")
            frames = []
            for t_idx in range(T):
                orig_frame = unnormalize(orig[:, t_idx, :, :])
                recon_frame = unnormalize(recon_b[:, t_idx, :, :])
                concat = torch.cat([orig_frame, recon_frame], dim=2)  # Side-by-side
                if concat.shape[2] > 512:  # Resize if too wide
                    concat = F.interpolate(concat.unsqueeze(0), size=(H, 512), mode='bilinear').squeeze(0)
                img = (concat.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                frames.append(img)

            imageio.mimsave(out_path, frames, fps=8)
            print(f"[INFO] Saved MP4 visualization: {out_path}")

    except Exception as e:
        msg = f"Failed to visualize reconstruction: {e}"
        if logger:
            logger.info(msg)
        else:
            print(f"[ERROR] {msg}")


def unnormalize_2(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """Revert normalization (ImageNet-style by default)."""
    mean = torch.tensor(mean, device=tensor.device)[None, :, None, None]
    std = torch.tensor(std, device=tensor.device)[None, :, None, None]
    return tensor * std + mean

def unnormalize_(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """Undo ImageNet normalization for visualization."""
    mean = torch.tensor(mean, device=tensor.device).view(-1, 1, 1)
    std = torch.tensor(std, device=tensor.device).view(-1, 1, 1)
    return tensor * std + mean

def visualize_reconstruction(
        clips, masks_enc, pred_pixel_values_list, patch_size, tubelet_size, in_chans, save_path, logger=None
):
    """
    Visualize reconstruction with multi-frame display.
    """
    try:
        import matplotlib.pyplot as plt
        from torchvision.utils import make_grid
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        device = clips.device
        batch_size = clips.shape[0]
        T, H, W = clips.shape[2], clips.shape[3], clips.shape[4]

        # Recreate patch grid
        patches = rearrange(
            clips,
            "b c (t p1) (h p2) (w p3) -> b (t h w) (p1 p2 p3 c)",
            p1=tubelet_size,
            p2=patch_size,
            p3=patch_size,
        )

        # Combine all mask indices into one set per sample
        mask_union = torch.zeros_like(patches[..., 0], dtype=torch.bool)
        for mask_block in masks_enc:
            batch_range = torch.arange(batch_size, device=device)[:, None]
            mask_union[batch_range, mask_block] = True

        # Merge predicted patches (ensure float dtype to avoid BF16 mismatch)
        reconstructed_patches = patches.float().clone()
        for pred_pixels, mask_block in zip(pred_pixel_values_list, masks_enc):
            batch_range = torch.arange(batch_size, device=device)[:, None]
            reconstructed_patches[batch_range, mask_block] = pred_pixels.float()
        # Move to CPU once after reconstruction
        reconstructed_patches = reconstructed_patches.cpu()
        mask_union = mask_union.cpu()
        clips = clips.cpu()

        # Reshape back to image
        recon_clips = rearrange(
            reconstructed_patches,
            "b (t h w) (p1 p2 p3 c) -> b c (t p1) (h p2) (w p3)",
            t=T // tubelet_size,
            h=H // patch_size,
            w=W // patch_size,
            p1=tubelet_size,
            p2=patch_size,
            p3=patch_size,
            c=in_chans,
        )

        # Select first sample
        b = 0
        orig = clips[b].detach().cpu()
        recon = recon_clips[b].detach().cpu()
        mask_union_b = mask_union[b].detach().cpu()

        # Convert union mask to spatial map
        t = T // tubelet_size
        h = H // patch_size
        w = W // patch_size
        mask_grid = mask_union_b.view(t, h, w).sum(dim=0) > 0

        # Select 4 representative frames
        frame_indices = np.arange(16) #[0, T // 3, 2 * T // 3, T - 1]
        num_frames_to_show = len(frame_indices)
        fig, axs = plt.subplots(3, num_frames_to_show, figsize=(num_frames_to_show * 3, 8))

        for i, frame_idx in enumerate(frame_indices):
            # orig_frame = orig[:, frame_idx, :, :]
            # recon_frame = recon[:, frame_idx, :, :]
            orig_frame = unnormalize(orig[:, frame_idx, :, :]).clamp(0, 1)
            recon_frame = unnormalize(recon[:, frame_idx, :, :]).clamp(0, 1)

            axs[0, i].imshow(orig_frame.permute(1, 2, 0))
            axs[0, i].set_title(f"Orig F{frame_idx}")
            axs[0, i].axis("off")

            axs[1, i].imshow(mask_grid.float(), cmap="gray")
            axs[1, i].set_title("Mask")
            axs[1, i].axis("off")

            axs[2, i].imshow(recon_frame.permute(1, 2, 0))
            axs[2, i].set_title("Recon")
            axs[2, i].axis("off")

        plt.tight_layout()
        plt.savefig(save_path, dpi=120)
        plt.close(fig)
        # if logger:
        #     logger.info(f"Saved reconstruction visualization: {save_path}")

    except Exception as e:
        if logger:
            logger.info(f"Failed to visualize reconstruction: {e}")


def merge_tokens(context, predicted, masks_ctxt, masks_tgt, num_patches):
    """
    Merge context (visible) and predicted (masked) features into full token grid.
    Handles list inputs and mixed-precision dtypes.
    """
    # Handle list inputs
    if isinstance(context, list):
        context = torch.cat(context, dim=1)
    if isinstance(predicted, list):
        predicted = torch.cat(predicted, dim=1)
    if isinstance(masks_ctxt, list):
        masks_ctxt = masks_ctxt[0]
    if isinstance(masks_tgt, list):
        masks_tgt = masks_tgt[0]

    B, _, D = context.shape
    device = context.device

    # Validate input shapes
    assert masks_ctxt.shape[1] == context.shape[1], f"Mismatch: masks_ctxt ({masks_ctxt.shape[1]}) vs context ({context.shape[1]})"
    assert masks_tgt.shape[1] == predicted.shape[1], f"Mismatch: masks_tgt ({masks_tgt.shape[1]}) vs predicted ({predicted.shape[1]})"

    # Prefer the higher precision dtype between context and predicted
    common_dtype = torch.promote_types(context.dtype, predicted.dtype)

    # Create merged tensor with the correct dtype
    merged = torch.zeros(B, num_patches, D, device=device, dtype=common_dtype)

    # Make sure both inputs match dtype
    context = context.to(common_dtype)
    predicted = predicted.to(common_dtype)

    batch_idx = torch.arange(B, device=device)[:, None]

    # Fill visible + masked tokens
    merged[batch_idx, masks_ctxt] = context
    merged[batch_idx, masks_tgt] = predicted

    return merged





