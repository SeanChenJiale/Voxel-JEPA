# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
import pdb
import sys
import warnings
import yaml


import torch

import src.models.vision_transformer as video_vit
import src.models.predictor as vit_pred
import src.models.decoder as decoder
from src.models.utils.multimask import MultiMaskWrapper, PredictorMultiMaskWrapper, DecoderMultiMaskWrapper
from src.utils.schedulers import (
    WarmupCosineSchedule,
    CosineWDSchedule)
from src.utils.tensors import trunc_normal_

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def load_checkpoint(
    r_path,
    encoder,
    predictor,
    target_encoder,
    opt,
    scaler,
    decoder=None,  # NEW: Add decoder parameter
    decoder_opt=None,  # NEW: Add decoder optimizer parameter
    decoder_scaler=None,  # NEW: Add decoder scaler parameter
):
    try:
        checkpoint = torch.load(r_path, map_location=torch.device('cpu'))
    except Exception as e:
        logger.info(f'Encountered exception when loading checkpoint {e}')

    epoch = 0
    try:
        epoch = checkpoint['epoch']

        # -- loading encoder
        pretrained_dict = checkpoint['encoder']
        msg = encoder.load_state_dict(pretrained_dict)
        logger.info(f'loaded pretrained encoder from epoch {epoch} with msg: {msg}')

        # -- loading predictor
        pretrained_dict = checkpoint['predictor']
        msg = predictor.load_state_dict(pretrained_dict)
        logger.info(f'loaded pretrained predictor from epoch {epoch} with msg: {msg}')

        # -- loading target_encoder
        if target_encoder is not None:
            print(list(checkpoint.keys()))
            pretrained_dict = checkpoint['target_encoder']
            msg = target_encoder.load_state_dict(pretrained_dict)
            logger.info(
                f'loaded pretrained target encoder from epoch {epoch} with msg: {msg}'
            )

        # -- loading optimizer
        opt.load_state_dict(checkpoint['opt'])
        if scaler is not None:
            scaler.load_state_dict(checkpoint['scaler'])
        logger.info(f'loaded optimizers from epoch {epoch}')
        
        # -- loading decoder (NEW)
        if decoder is not None and 'decoder' in checkpoint and checkpoint['decoder'] is not None:
            pretrained_dict = checkpoint['decoder']
            msg = decoder.load_state_dict(pretrained_dict)
            logger.info(f'loaded pretrained decoder from epoch {epoch} with msg: {msg}')
        elif decoder is not None:
            logger.info('No decoder checkpoint found, using randomly initialized decoder')
        
        # -- loading decoder optimizer (NEW)
        if decoder_opt is not None and 'decoder_opt' in checkpoint and checkpoint['decoder_opt'] is not None:
            decoder_opt.load_state_dict(checkpoint['decoder_opt'])
            logger.info(f'loaded decoder optimizer from epoch {epoch}')
        elif decoder_opt is not None:
            logger.info('No decoder optimizer checkpoint found, using randomly initialized decoder optimizer')
            
        # -- loading decoder scaler (NEW)
        if decoder_scaler is not None and 'decoder_scaler' in checkpoint and checkpoint['decoder_scaler'] is not None:
            decoder_scaler.load_state_dict(checkpoint['decoder_scaler'])
            logger.info(f'loaded decoder scaler from epoch {epoch}')
        elif decoder_scaler is not None:
            logger.info('No decoder scaler checkpoint found, using randomly initialized decoder scaler')
        
        logger.info(f'read-path: {r_path}')
        del checkpoint

    except Exception as e:
        logger.info(f'Encountered exception when loading checkpoint {e}')
        epoch = 0

    return (
        encoder,
        predictor,
        target_encoder,
        opt,
        scaler,
        epoch,
    )


def init_video_model(
    device,
    patch_size=16,
    num_frames=16,
    tubelet_size=2,
    model_name='vit_base',
    crop_size=224,
    pred_depth=6,
    pred_embed_dim=384,
    uniform_power=False,
    use_mask_tokens=False,
    num_mask_tokens=2,
    zero_init_mask_tokens=True,
    use_sdpa=False,
):
    encoder = video_vit.__dict__[model_name](
        img_size=crop_size,
        patch_size=patch_size,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        uniform_power=uniform_power,
        use_sdpa=use_sdpa,
    )
    print(f"pred_embed_dim: {pred_embed_dim}")
    encoder = MultiMaskWrapper(encoder)
    predictor = vit_pred.__dict__['vit_predictor'](
        img_size=crop_size,
        use_mask_tokens=use_mask_tokens,
        patch_size=patch_size,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        embed_dim=encoder.backbone.embed_dim,
        predictor_embed_dim=pred_embed_dim,
        depth=pred_depth,
        num_heads=encoder.backbone.num_heads,
        uniform_power=uniform_power,
        num_mask_tokens=num_mask_tokens,
        zero_init_mask_tokens=zero_init_mask_tokens,
        use_sdpa=use_sdpa,
    )
    predictor = PredictorMultiMaskWrapper(predictor)

    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)

    for m in encoder.modules():
        init_weights(m)

    for m in predictor.modules():
        init_weights(m)

    encoder.to(device)
    predictor.to(device)
    logger.info(encoder)
    logger.info(predictor)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f'Encoder number of parameters: {count_parameters(encoder)}')
    logger.info(f'Predictor number of parameters: {count_parameters(predictor)}')

    return encoder, predictor


def init_decoder(
    device,
    patch_size=16,
    num_frames=16,
    tubelet_size=2,
    model_name='vit_base',
    crop_size=224,
    decoder_type='linear',
    decoder_depth=4,
    decoder_heads=12,
    decoder_dim=512,
    in_chans=3,
):
    """
    Initialize decoder for pixel reconstruction
    
    Args:
        device: Device to place decoder on
        patch_size: Size of patches
        num_frames: Number of frames in video
        tubelet_size: Size of tubelets
        model_name: Name of the base model
        crop_size: Size of input crop
        decoder_type: Type of decoder ('pixel' or 'conv')
        decoder_depth: Depth of decoder transformer
        decoder_heads: Number of attention heads
        decoder_dim: Dimension of decoder
        in_chans: Number of input channels
    
    Returns:
        decoder: Initialized decoder model
    """
    
    # Get encoder embed dimension based on model name
    embed_dims = {
        'vit_tiny': 192,
        'vit_small': 384,
        'vit_base': 768,
        'vit_large': 1024,
        'vit_huge': 1280,
        'vit_giant': 1408,
        'vit_gigantic': 1664,
    }
    
    encoder_embed_dim = embed_dims.get(model_name, 768)

    decoder_model = decoder.VJEPADecoder(
        encoder_embed_dim=encoder_embed_dim,
        patch_size=patch_size,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        in_chans=in_chans,
        decoder_type=decoder_type,
        decoder_depth=decoder_depth,
        decoder_heads=decoder_heads,
        decoder_dim=decoder_dim,
    )
    
    decoder_model.to(device)
    logger.info(f'Decoder initialized: {decoder_type} type')
    
    # Wrap decoder to handle multiple mask blocks
    # decoder_model = DecoderMultiMaskWrapper(decoder_model)
    
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f'Decoder number of parameters: {count_parameters(decoder_model)}')
    
    return decoder_model


def init_opt(
        encoder,
        predictor,
        decoder,  # NEW: Add decoder parameter (optional)
        iterations_per_epoch,
        start_lr,
        ref_lr,
        warmup,
        num_epochs,
        wd=1e-6,
        final_wd=1e-6,
        final_lr=0.0,
        mixed_precision=False,
        ipe_scale=1.25,
        betas=(0.9, 0.999),
        eps=1e-8,
        zero_init_bias_wd=True,
):
    param_groups = []
    
    # Add encoder parameters if encoder is provided
    if encoder is not None:
        param_groups.extend([
            {
                'params': (p for n, p in encoder.named_parameters()
                           if ('bias' not in n) and (len(p.shape) != 1))
            }, {
                'params': (p for n, p in encoder.named_parameters()
                           if ('bias' in n) or (len(p.shape) == 1)),
                'WD_exclude': zero_init_bias_wd,
                'weight_decay': 0,
            }
        ])
    
    # Add predictor parameters if predictor is provided
    if predictor is not None:
        param_groups.extend([
            {
                'params': (p for n, p in predictor.named_parameters()
                           if ('bias' not in n) and (len(p.shape) != 1))
            }, {
                'params': (p for n, p in predictor.named_parameters()
                           if ('bias' in n) or (len(p.shape) == 1)),
                'WD_exclude': zero_init_bias_wd,
                'weight_decay': 0,
            }
        ])
    
    # Add decoder parameters if decoder is provided
    if decoder is not None:
        param_groups.extend([
            {
                'params': (p for n, p in decoder.named_parameters()
                           if ('bias' not in n) and (len(p.shape) != 1))
            }, {
                'params': (p for n, p in decoder.named_parameters()
                           if ('bias' in n) or (len(p.shape) == 1)),
                'WD_exclude': zero_init_bias_wd,
                'weight_decay': 0,
            }
        ])

    logger.info('Using AdamW')
    optimizer = torch.optim.AdamW(param_groups, betas=betas, eps=eps)
    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=int(warmup * iterations_per_epoch),
        start_lr=start_lr,
        ref_lr=ref_lr,
        final_lr=final_lr,
        T_max=int(ipe_scale * num_epochs * iterations_per_epoch),
    )
    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=wd,
        final_wd=final_wd,
        T_max=int(ipe_scale * num_epochs * iterations_per_epoch),
    )
    scaler = torch.cuda.amp.GradScaler() if mixed_precision else None
    return optimizer, scaler, scheduler, wd_scheduler

def init_opt_(
    encoder,
    predictor,
    iterations_per_epoch,
    start_lr,
    ref_lr,
    warmup,
    num_epochs,
    wd=1e-6,
    final_wd=1e-6,
    final_lr=0.0,
    mixed_precision=False,
    ipe_scale=1.25,
    betas=(0.9, 0.999),
    eps=1e-8,
    zero_init_bias_wd=True,
):
    param_groups = [
        {
            'params': (p for n, p in encoder.named_parameters()
                       if ('bias' not in n) and (len(p.shape) != 1))
        }, {
            'params': (p for n, p in predictor.named_parameters()
                       if ('bias' not in n) and (len(p.shape) != 1))
        }, {
            'params': (p for n, p in encoder.named_parameters()
                       if ('bias' in n) or (len(p.shape) == 1)),
            'WD_exclude': zero_init_bias_wd,
            'weight_decay': 0,
        }, {
            'params': (p for n, p in predictor.named_parameters()
                       if ('bias' in n) or (len(p.shape) == 1)),
            'WD_exclude': zero_init_bias_wd,
            'weight_decay': 0,
        },
    ]

    logger.info('Using AdamW')
    optimizer = torch.optim.AdamW(param_groups, betas=betas, eps=eps)
    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=int(warmup * iterations_per_epoch),
        start_lr=start_lr,
        ref_lr=ref_lr,
        final_lr=final_lr,
        T_max=int(ipe_scale * num_epochs * iterations_per_epoch),
    )
    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=wd,
        final_wd=final_wd,
        T_max=int(ipe_scale * num_epochs * iterations_per_epoch),
    )
    scaler = torch.cuda.amp.GradScaler() if mixed_precision else None
    return optimizer, scaler, scheduler, wd_scheduler
