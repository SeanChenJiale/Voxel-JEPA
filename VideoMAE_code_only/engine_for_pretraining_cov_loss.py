import math
import sys
from typing import Iterable
import torch
import torch.nn as nn
import utils
from einops import rearrange
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from entropy_loss import calculate_trace_torch

def train_one_epoch(model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0, patch_size: int = 16, 
                    normlize_target: bool = True, log_writer=None, lr_scheduler=None, start_steps=None,
                    lr_schedule_values=None, wd_schedule_values=None, cov_loss_weight=0.1, atlas_list=None, feature_type='entropy',
                    use_trace_sum=True, output_dir=None, n_parameters=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    loss_func = nn.MSELoss()

    for step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # assign learning rate & weight decay for each step
        it = start_steps + step  # global training iteration
        if lr_schedule_values is not None or wd_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        videos, bool_masked_pos, axis, indices = batch
        videos = videos.to(device, non_blocking=True)
        bool_masked_pos = bool_masked_pos.to(device, non_blocking=True).flatten(1).to(torch.bool)
        axis = axis.to(device, non_blocking=True)
        indices = indices.to(device, non_blocking=True)

        with torch.no_grad():
            # calculate the predict label
            mean = torch.as_tensor([0.146689, 0.146689, 0.146689]).to(device)[None, :, None, None, None]
            std = torch.as_tensor([0.267249, 0.267249, 0.267249]).to(device)[None, :, None, None, None]
            unnorm_videos = videos * std + mean  # in [0, 1]

            if normlize_target:
                videos_squeeze = rearrange(unnorm_videos, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2) c', p0=2, p1=patch_size, p2=patch_size)
                videos_norm = (videos_squeeze - videos_squeeze.mean(dim=-2, keepdim=True)
                    ) / (videos_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
                # we find that the mean is about 0.48 and standard deviation is about 0.08.
                videos_patch = rearrange(videos_norm, 'b n p c -> b n (p c)')
            else:
                videos_patch = rearrange(unnorm_videos, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2 c)', p0=2, p1=patch_size, p2=patch_size)

            B, _, C = videos_patch.shape
            labels = videos_patch[bool_masked_pos].reshape(B, -1, C)

            orig_trace = calculate_trace_torch(unnorm_videos, indices, axis, device=videos.device, atlas_list=atlas_list, feature_type=feature_type, use_trace_sum=use_trace_sum) 

        with torch.cuda.amp.autocast():
            outputs = model(videos, bool_masked_pos)
            mae_loss = loss_func(input=outputs, target=labels)

            # Use float16 for reconstruction to speed up computation (keep dtype consistency with autocast)
            full_patches = videos_patch.clone().to(torch.float16)
            full_patches[bool_masked_pos] = outputs.flatten(0, 1)  # outputs is already float16 from autocast

            reconstructed_video = rearrange(full_patches, 'b n (p c) -> b n p c', c=3)
            reconstructed_video = reconstructed_video * (videos_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6) + videos_squeeze.mean(dim=-2, keepdim=True)
            reconstructed_video = rearrange(reconstructed_video, 'b (t h w) (p0 p1 p2) c -> b c (t p0) (h p1) (w p2)', p0=2, p1=patch_size, p2=patch_size, h=14, w=14)

            recon_trace = calculate_trace_torch(reconstructed_video, indices, axis, device=outputs.device, atlas_list=atlas_list, feature_type=feature_type, use_trace_sum=use_trace_sum)
            
            # Check for NaN/Inf in traces before computing loss
            # Handle both scalar and tensor cases
            if use_trace_sum:
                # Scalar case
                if not torch.isfinite(orig_trace):
                    print(f"Warning: NaN/Inf detected in orig_trace: {orig_trace}")
                    orig_trace = torch.tensor(0.0, device=orig_trace.device, dtype=orig_trace.dtype)
                
                if not torch.isfinite(recon_trace):
                    print(f"Warning: NaN/Inf detected in recon_trace: {recon_trace}")
                    recon_trace = torch.tensor(0.0, device=recon_trace.device, dtype=recon_trace.dtype)
                
                cov_loss = torch.abs(orig_trace - recon_trace)
            else:
                # Tensor case (original behavior)
                if not torch.all(torch.isfinite(orig_trace)):
                    print(f"Warning: NaN/Inf detected in orig_trace: {orig_trace}")
                    orig_trace = torch.where(torch.isfinite(orig_trace), orig_trace, torch.zeros_like(orig_trace))
                
                if not torch.all(torch.isfinite(recon_trace)):
                    print(f"Warning: NaN/Inf detected in recon_trace: {recon_trace}")
                    recon_trace = torch.where(torch.isfinite(recon_trace), recon_trace, torch.zeros_like(recon_trace))
                
                cov_loss = nn.L1Loss()(orig_trace, recon_trace)
        
        mae_loss_value = mae_loss.item()
        cov_loss_value = cov_loss.item()

        if not math.isfinite(mae_loss_value):
            print("Loss is {}, stopping training".format(mae_loss_value))
            sys.exit(1)
        
        if not math.isfinite(cov_loss_value):
            print("Cov Loss is {}, stopping training".format(cov_loss_value))
            print(f"orig_trace: {orig_trace}")
            print(f"recon_trace: {recon_trace}")
            sys.exit(1)

        loss = mae_loss + cov_loss_weight * cov_loss

        optimizer.zero_grad()
        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                parameters=model.parameters(), create_graph=is_second_order)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        metric_logger.update(mae_loss=mae_loss_value)
        metric_logger.update(cov_loss=cov_loss_value)
        # Handle both scalar and tensor cases for trace logging
        if use_trace_sum:
            metric_logger.update(orig_trace=orig_trace.item() if torch.is_tensor(orig_trace) else orig_trace)
            metric_logger.update(recon_trace=recon_trace.item() if torch.is_tensor(recon_trace) else recon_trace)
        else:
            metric_logger.update(orig_trace=orig_trace.mean().item())
            metric_logger.update(recon_trace=recon_trace.mean().item())
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)

        if log_writer is not None:
            log_writer.update(mae_loss=mae_loss_value, head="loss")
            log_writer.update(cov_loss=cov_loss_value, head="loss")
            # Handle both scalar and tensor cases for trace logging
            if use_trace_sum:
                log_writer.update(orig_trace=orig_trace.item() if torch.is_tensor(orig_trace) else orig_trace, head="loss")
                log_writer.update(recon_trace=recon_trace.item() if torch.is_tensor(recon_trace) else recon_trace, head="loss")
            else:
                log_writer.update(orig_trace=orig_trace.mean().item(), head="loss")
                log_writer.update(recon_trace=recon_trace.mean().item(), head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.set_step()

        # Save statistics to log.txt every print_freq iterations
        if output_dir is not None and utils.is_main_process() and step % print_freq == 0:
            import json
            import os
            it = start_steps + step
            # Handle both scalar and tensor cases for trace logging
            if use_trace_sum:
                orig_trace_value = orig_trace.item() if torch.is_tensor(orig_trace) else orig_trace
                recon_trace_value = recon_trace.item() if torch.is_tensor(recon_trace) else recon_trace
            else:
                orig_trace_value = orig_trace.mean().item()
                recon_trace_value = recon_trace.mean().item()
            
            log_stats = {
                'train_lr': max_lr,
                'train_min_lr': min_lr,
                'train_mae_loss': mae_loss_value,
                'train_cov_loss': cov_loss_value,
                'train_orig_trace': orig_trace_value,
                'train_recon_trace': recon_trace_value,
                'train_loss_scale': loss_scale_value,
                'train_weight_decay': weight_decay_value if weight_decay_value is not None else 0.0,
                'iteration': it,
                'epoch': epoch,
                'step': step,
                'n_parameters': n_parameters if n_parameters is not None else 0
            }
            with open(os.path.join(output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
