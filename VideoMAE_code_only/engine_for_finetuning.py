import os
import numpy as np
import math
import sys
from typing import Iterable, Optional
import torch
from mixup import Mixup
from timm.utils import accuracy, ModelEma
import utils
from scipy.special import softmax
from sklearn.metrics import roc_auc_score, f1_score
from scipy.stats import pearsonr


def gather_distributed_outputs_targets(all_outputs_list, all_targets_list, device, task_type='classification'):
    """
    Gather outputs and targets from all processes in distributed training.
    
    Args:
        all_outputs_list: List of numpy arrays (outputs from batches on this process)
        all_targets_list: List of numpy arrays (targets from batches on this process)
        device: Device to use for tensor operations
        task_type: 'classification' or 'regression'
    
    Returns:
        all_outputs_gathered: Concatenated outputs from all processes (numpy array, only on rank 0)
        all_targets_gathered: Concatenated targets from all processes (numpy array, only on rank 0)
    """
    # Concatenate local outputs and targets first
    local_outputs = np.concatenate(all_outputs_list, axis=0) if len(all_outputs_list) > 0 else np.array([])
    local_targets = np.concatenate(all_targets_list, axis=0) if len(all_targets_list) > 0 else np.array([])
    
    if not utils.is_dist_avail_and_initialized():
        # Single process case
        return local_outputs, local_targets
    
    # Get world size
    world_size = utils.get_world_size()
    
    # Get local sizes first to determine max_size and num_classes
    local_size = local_outputs.shape[0] if local_outputs.size > 0 else 0
    if task_type == 'regression':
        local_num_classes = 1  # Regression outputs are 1D
    else:
        local_num_classes = local_outputs.shape[1] if local_outputs.size > 0 and len(local_outputs.shape) > 1 else 0
    
    # Gather sizes and whether each process has data
    sizes_tensor = torch.tensor([local_size], dtype=torch.long, device=device)
    num_classes_tensor = torch.tensor([local_num_classes], dtype=torch.long, device=device)
    
    sizes_list = [torch.tensor([0], dtype=torch.long, device=device) for _ in range(world_size)]
    num_classes_list = [torch.tensor([0], dtype=torch.long, device=device) for _ in range(world_size)]
    
    torch.distributed.all_gather(sizes_list, sizes_tensor)
    torch.distributed.all_gather(num_classes_list, num_classes_tensor)
    
    sizes = [int(s.item()) for s in sizes_list]
    num_classes_list = [int(nc.item()) for nc in num_classes_list]
    max_size = max(sizes) if sizes else 0
    
    # Handle empty case
    if max_size == 0:
        if utils.is_main_process():
            return np.array([]), np.array([])
        else:
            return None, None
    
    # Get num_classes from first process that has data
    num_classes = next((nc for nc in num_classes_list if nc > 0), 2)
    
    # Convert local arrays to tensors
    if local_outputs.size > 0:
        local_outputs_tensor = torch.from_numpy(local_outputs).to(device)
        local_targets_tensor = torch.from_numpy(local_targets).to(device)
        if task_type == 'regression':
            # Ensure outputs are 1D for regression
            if len(local_outputs_tensor.shape) > 1:
                local_outputs_tensor = local_outputs_tensor.squeeze(-1)
            local_targets_tensor = local_targets_tensor.float()
        else:
            local_targets_tensor = local_targets_tensor.long()
    else:
        # Empty case - create dummy tensors with correct shape
        if task_type == 'regression':
            local_outputs_tensor = torch.empty((0,), device=device, dtype=torch.float32)
            local_targets_tensor = torch.empty((0,), device=device, dtype=torch.float32)
        else:
            local_outputs_tensor = torch.empty((0, num_classes), device=device, dtype=torch.float32)
            local_targets_tensor = torch.empty((0,), device=device, dtype=torch.long)
    
    # Pad tensors to max_size if needed
    if local_outputs_tensor.shape[0] < max_size:
        if task_type == 'regression':
            padding_outputs = torch.zeros(max_size - local_outputs_tensor.shape[0], 
                                         dtype=local_outputs_tensor.dtype, device=device)
            padding_targets = torch.zeros(max_size - local_targets_tensor.shape[0], 
                                         dtype=local_targets_tensor.dtype, device=device)
        else:
            padding_outputs = torch.zeros(max_size - local_outputs_tensor.shape[0], num_classes, 
                                         dtype=local_outputs_tensor.dtype, device=device)
            padding_targets = torch.zeros(max_size - local_targets_tensor.shape[0], 
                                         dtype=local_targets_tensor.dtype, device=device)
        local_outputs_tensor = torch.cat([local_outputs_tensor, padding_outputs])
        local_targets_tensor = torch.cat([local_targets_tensor, padding_targets])
    
    # Gather tensors from all processes
    gathered_outputs = [torch.zeros_like(local_outputs_tensor) for _ in range(world_size)]
    gathered_targets = [torch.zeros_like(local_targets_tensor) for _ in range(world_size)]
    
    torch.distributed.all_gather(gathered_outputs, local_outputs_tensor)
    torch.distributed.all_gather(gathered_targets, local_targets_tensor)
    
    # Concatenate and trim to actual sizes, then convert back to numpy on rank 0
    if utils.is_main_process():
        outputs_list = []
        targets_list = []
        for i in range(world_size):
            if sizes[i] > 0:
                outputs_list.append(gathered_outputs[i][:sizes[i]].cpu().numpy())
                targets_list.append(gathered_targets[i][:sizes[i]].cpu().numpy())
        all_outputs_gathered = np.concatenate(outputs_list, axis=0) if outputs_list else np.array([])
        all_targets_gathered = np.concatenate(targets_list, axis=0) if targets_list else np.array([])
        return all_outputs_gathered, all_targets_gathered
    else:
        return None, None


def train_class_batch(model, samples, target, criterion, task_type='classification'):
    outputs = model(samples)
    if task_type == 'regression':
        # For regression, outputs is (B, 1), squeeze to (B,)
        outputs = outputs.squeeze(1)
        target = target.float()
    loss = criterion(outputs, target)
    return loss, outputs


def get_loss_scale_for_deepspeed(model):
    optimizer = model.optimizer
    return optimizer.loss_scale if hasattr(optimizer, "loss_scale") else optimizer.cur_scale


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None, log_writer=None,
                    start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None,
                    output_dir=None, n_parameters=None, task_type='classification'):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()

    for data_iter_step, (samples, targets, _, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None and task_type == 'classification':
            samples, targets = mixup_fn(samples, targets)

        if loss_scaler is None:
            samples = samples.half()
            loss, output = train_class_batch(
                model, samples, targets, criterion, task_type)
        else:
            with torch.cuda.amp.autocast():
                loss, output = train_class_batch(
                    model, samples, targets, criterion, task_type)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        if loss_scaler is None:
            loss /= update_freq
            model.backward(loss)
            model.step()

            if (data_iter_step + 1) % update_freq == 0:
                # model.zero_grad()
                # Deepspeed will call step() & model.zero_grad() automatic
                if model_ema is not None:
                    model_ema.update(model)
            grad_norm = None
            loss_scale_value = get_loss_scale_for_deepspeed(model)
        else:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
            loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        if task_type == 'classification':
            if mixup_fn is None:
                class_acc = (output.max(-1)[-1] == targets).float().mean()
            else:
                class_acc = None
            metric_logger.update(class_acc=class_acc)
        else:
            class_acc = None
        metric_logger.update(loss=loss_value)
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
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(class_acc=class_acc, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")

            log_writer.set_step()

        # Save statistics to log.txt every print_freq iterations
        if output_dir is not None and utils.is_main_process() and data_iter_step % print_freq == 0:
            import json
            import os
            log_stats = {
                'train_lr': max_lr,
                'train_min_lr': min_lr,
                'train_loss': loss_value,
                'train_class_acc': class_acc.item() if class_acc is not None else None,
                'train_loss_scale': loss_scale_value,
                'train_weight_decay': weight_decay_value,
                'iteration': it,
                'epoch': epoch,
                'data_iter_step': data_iter_step,
                'step': step,
                'n_parameters': n_parameters if n_parameters is not None else 0
            }
            with open(os.path.join(output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validation_one_epoch(data_loader, model, device, task_type='classification', num_classes=2):
    if task_type == 'regression':
        criterion = torch.nn.L1Loss()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Val:'

    # switch to evaluation mode
    model.eval()

    all_outputs = []
    all_targets = []
    
    for batch in metric_logger.log_every(data_loader, 10, header):
        videos = batch[0]
        target = batch[1]
        videos = videos.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(videos)
            if task_type == 'regression':
                output = output.squeeze(1)
                target = target.float()
            loss = criterion(output, target)

        batch_size = videos.shape[0]
        metric_logger.update(loss=loss.item())
        
        if task_type == 'classification':
            # acc1, acc5 = accuracy(output, target, topk=(1, 5))
            acc1 = accuracy(output, target, topk=(1, ))[0]
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            # Collect outputs and targets for AUROC calculation
            all_outputs.append(torch.softmax(output, dim=1).cpu().numpy())
            all_targets.append(target.cpu().numpy())
        else:
            # Regression: collect raw outputs and targets
            all_outputs.append(output.cpu().numpy())
            all_targets.append(target.cpu().numpy())
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    
    # Gather outputs and targets from all processes
    all_outputs_gathered, all_targets_gathered = gather_distributed_outputs_targets(
        all_outputs, all_targets, device, task_type=task_type
    )
    
    if task_type == 'regression':
        # Calculate MAE and Pearson correlation for regression
        mae = 0.0
        pearson = 0.0
        if not utils.is_dist_avail_and_initialized() or utils.is_main_process():
            if all_outputs_gathered is not None and all_outputs_gathered.size > 0 and all_targets_gathered.size > 0:
                # Flatten arrays for regression
                all_outputs_flat = all_outputs_gathered.flatten()
                all_targets_flat = all_targets_gathered.flatten()
                mae = np.mean(np.abs(all_outputs_flat - all_targets_flat))
                try:
                    pearson, _ = pearsonr(all_outputs_flat, all_targets_flat)
                except:
                    pearson = 0.0
                print('* MAE {mae:.4f} Pearson {pearson:.4f} loss {losses.global_avg:.3f}'
                      .format(mae=mae, pearson=pearson, losses=metric_logger.loss))
            else:
                print('* loss {losses.global_avg:.3f}'.format(losses=metric_logger.loss))
        
        result = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        result['mae'] = float(mae)  # Convert numpy type to Python float
        result['pearson'] = float(pearson)  # Convert numpy type to Python float
        return result
    else:
        # Calculate AUROC and F1 using gathered data (only on main process in distributed case)
        auroc = 0.0
        f1 = 0.0
        if not utils.is_dist_avail_and_initialized() or utils.is_main_process():
            if all_outputs_gathered is not None and all_outputs_gathered.size > 0 and all_targets_gathered.size > 0:
                # Calculate predictions for F1 score
                all_preds_gathered = np.argmax(all_outputs_gathered, axis=1)
                
                try:
                    # Normalize probabilities to ensure they sum to exactly 1.0 (fix floating point precision issues)
                    prob_sums = all_outputs_gathered.sum(axis=1, keepdims=True)
                    all_outputs_gathered_normalized = all_outputs_gathered / (prob_sums + 1e-10)
                    
                    if all_outputs_gathered.shape[1] == 2:  # Binary classification
                        auroc = roc_auc_score(all_targets_gathered, all_outputs_gathered_normalized[:, 1])
                        f1 = f1_score(all_targets_gathered, all_preds_gathered, average='binary')
                    else:  # Multi-class classification
                        auroc = roc_auc_score(all_targets_gathered, all_outputs_gathered_normalized, multi_class='ovr', average='macro')
                        f1 = f1_score(all_targets_gathered, all_preds_gathered, average='weighted')
                    print('* Acc@1 {top1.global_avg:.3f} AUROC {auroc:.3f} F1 {f1:.3f} loss {losses.global_avg:.3f}'
                          .format(top1=metric_logger.acc1, auroc=auroc, f1=f1, losses=metric_logger.loss))
                except ValueError as e:
                    # More informative error message
                    print(f"ValueError in AUROC/F1 calculation: {str(e)}")
                    print(f"all_outputs_gathered shape: {all_outputs_gathered.shape}")
                    print(f"all_targets_gathered shape: {all_targets_gathered.shape}")
                    
                    # Check probability sums
                    prob_sums = all_outputs_gathered.sum(axis=1)
                    print(f"Probability sum range: [{prob_sums.min():.6f}, {prob_sums.max():.6f}]")
                    print(f"Rows not summing to 1.0: {(np.abs(prob_sums - 1.0) > 1e-5).sum()}")
                    
                    # Try with normalized probabilities
                    try:
                        prob_sums = all_outputs_gathered.sum(axis=1, keepdims=True)
                        all_outputs_gathered_normalized = all_outputs_gathered / (prob_sums + 1e-10)
                        
                        if all_outputs_gathered.shape[1] == 2:
                            auroc = roc_auc_score(all_targets_gathered, all_outputs_gathered_normalized[:, 1])
                        else:
                            auroc = roc_auc_score(all_targets_gathered, all_outputs_gathered_normalized, multi_class='ovr', average='macro')
                        f1 = f1_score(all_targets_gathered, all_preds_gathered, 
                                     average='binary' if all_outputs_gathered.shape[1] == 2 else 'weighted')
                        print('* Acc@1 {top1.global_avg:.3f} AUROC {auroc:.3f} F1 {f1:.3f} loss {losses.global_avg:.3f}'
                              .format(top1=metric_logger.acc1, auroc=auroc, f1=f1, losses=metric_logger.loss))
                    except ValueError:
                        # Calculate F1 even if AUROC fails
                        try:
                            f1 = f1_score(all_targets_gathered, all_preds_gathered, 
                                         average='binary' if all_outputs_gathered.shape[1] == 2 else 'weighted')
                            print('* Acc@1 {top1.global_avg:.3f} F1 {f1:.3f} loss {losses.global_avg:.3f}'
                                  .format(top1=metric_logger.acc1, f1=f1, losses=metric_logger.loss))
                        except ValueError:
                            print('* Acc@1 {top1.global_avg:.3f} loss {losses.global_avg:.3f}'
                                  .format(top1=metric_logger.acc1, losses=metric_logger.loss))
            else:
                print('* Acc@1 {top1.global_avg:.3f} loss {losses.global_avg:.3f}'
                      .format(top1=metric_logger.acc1, losses=metric_logger.loss))
        
        result = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        result['auroc'] = auroc
        result['f1'] = f1
        return result



@torch.no_grad()
def final_test(data_loader, model, device, task_type='classification', num_classes=2):
    if task_type == 'regression':
        criterion = torch.nn.L1Loss()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    all_outputs = []
    all_targets = []
    
    for batch in metric_logger.log_every(data_loader, 10, header):
        videos = batch[0]
        target = batch[1]
        videos = videos.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(videos)
            if task_type == 'regression':
                output = output.squeeze(1)
                target = target.float()
            loss = criterion(output, target)

        batch_size = videos.shape[0]
        metric_logger.update(loss=loss.item())
        
        if task_type == 'classification':
            acc1 = accuracy(output, target, topk=(1, ))[0]
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            all_outputs.append(torch.softmax(output, dim=1).cpu().numpy())
            all_targets.append(target.cpu().numpy())
        else:
            # Regression: collect raw outputs and targets
            all_outputs.append(output.cpu().numpy())
            all_targets.append(target.cpu().numpy())
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    
    # Gather outputs and targets from all processes
    all_outputs_gathered, all_targets_gathered = gather_distributed_outputs_targets(
        all_outputs, all_targets, device, task_type=task_type
    )

    if task_type == 'regression':
        # Calculate MAE and Pearson correlation for regression
        mae = 0.0
        pearson = 0.0
        if not utils.is_dist_avail_and_initialized() or utils.is_main_process():
            if all_outputs_gathered is not None and all_outputs_gathered.size > 0 and all_targets_gathered.size > 0:
                # Flatten arrays for regression
                all_outputs_flat = all_outputs_gathered.flatten()
                all_targets_flat = all_targets_gathered.flatten()
                mae = np.mean(np.abs(all_outputs_flat - all_targets_flat))
                try:
                    pearson, _ = pearsonr(all_outputs_flat, all_targets_flat)
                except:
                    pearson = 0.0
                print('* MAE {mae:.4f} Pearson {pearson:.4f} loss {losses.global_avg:.3f}'
                      .format(mae=mae, pearson=pearson, losses=metric_logger.loss))
            else:
                print('* loss {losses.global_avg:.3f}'.format(losses=metric_logger.loss))
        
        result = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        result['mae'] = float(mae)  # Convert numpy type to Python float
        result['pearson'] = float(pearson)  # Convert numpy type to Python float
        return result
    else:
        # Calculate AUROC and F1 using gathered data (only on main process in distributed case)
        auroc = 0.0
        f1 = 0.0
        if not utils.is_dist_avail_and_initialized() or utils.is_main_process():
            if all_outputs_gathered is not None and all_outputs_gathered.size > 0 and all_targets_gathered.size > 0:
                # Calculate predictions for F1 score
                all_preds_gathered = np.argmax(all_outputs_gathered, axis=1)
                
                try:
                    # Normalize probabilities to ensure they sum to exactly 1.0 (fix floating point precision issues)
                    prob_sums = all_outputs_gathered.sum(axis=1, keepdims=True)
                    all_outputs_gathered_normalized = all_outputs_gathered / (prob_sums + 1e-10)
                    
                    if all_outputs_gathered.shape[1] == 2:  # Binary classification
                        if utils.is_main_process():
                            print(f"Gathered outputs shape: {all_outputs_gathered.shape}")
                            print(f"Gathered targets shape: {all_targets_gathered.shape}")
                        auroc = roc_auc_score(all_targets_gathered, all_outputs_gathered_normalized[:, 1])
                        f1 = f1_score(all_targets_gathered, all_preds_gathered, average='binary')
                    else:  # Multi-class classification
                        auroc = roc_auc_score(all_targets_gathered, all_outputs_gathered_normalized, multi_class='ovr', average='macro')
                        f1 = f1_score(all_targets_gathered, all_preds_gathered, average='weighted')
                    print('* Acc@1 {top1.global_avg:.3f} AUROC {auroc:.3f} F1 {f1:.3f} loss {losses.global_avg:.3f}'
                          .format(top1=metric_logger.acc1, auroc=auroc, f1=f1, losses=metric_logger.loss))
                except ValueError as e:
                    # More informative error message
                    print(f"ValueError in AUROC/F1 calculation: {str(e)}")
                    print(f"all_outputs_gathered shape: {all_outputs_gathered.shape}")
                    print(f"all_targets_gathered shape: {all_targets_gathered.shape}")
                    
                    # Check probability sums
                    prob_sums = all_outputs_gathered.sum(axis=1)
                    print(f"Probability sum range: [{prob_sums.min():.6f}, {prob_sums.max():.6f}]")
                    print(f"Rows not summing to 1.0: {(np.abs(prob_sums - 1.0) > 1e-5).sum()}")
                    
                    # Try with normalized probabilities
                    try:
                        prob_sums = all_outputs_gathered.sum(axis=1, keepdims=True)
                        all_outputs_gathered_normalized = all_outputs_gathered / (prob_sums + 1e-10)
                        
                        if all_outputs_gathered.shape[1] == 2:
                            auroc = roc_auc_score(all_targets_gathered, all_outputs_gathered_normalized[:, 1])
                        else:
                            auroc = roc_auc_score(all_targets_gathered, all_outputs_gathered_normalized, multi_class='ovr', average='macro')
                        f1 = f1_score(all_targets_gathered, all_preds_gathered, 
                                     average='binary' if all_outputs_gathered.shape[1] == 2 else 'weighted')
                        print('* Acc@1 {top1.global_avg:.3f} AUROC {auroc:.3f} F1 {f1:.3f} loss {losses.global_avg:.3f}'
                              .format(top1=metric_logger.acc1, auroc=auroc, f1=f1, losses=metric_logger.loss))
                    except ValueError:
                        # Calculate F1 even if AUROC fails
                        try:
                            f1 = f1_score(all_targets_gathered, all_preds_gathered, 
                                         average='binary' if all_outputs_gathered.shape[1] == 2 else 'weighted')
                            print('* Acc@1 {top1.global_avg:.3f} F1 {f1:.3f} loss {losses.global_avg:.3f}'
                                  .format(top1=metric_logger.acc1, f1=f1, losses=metric_logger.loss))
                        except ValueError:
                            print('* Acc@1 {top1.global_avg:.3f} loss {losses.global_avg:.3f}'
                                  .format(top1=metric_logger.acc1, losses=metric_logger.loss))
            else:
                print('* Acc@1 {top1.global_avg:.3f} loss {losses.global_avg:.3f}'
                      .format(top1=metric_logger.acc1, losses=metric_logger.loss))
        
        result = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        result['auroc'] = auroc
        result['f1'] = f1
        return result


def merge(eval_path, num_tasks):
    dict_feats = {}
    dict_label = {}
    dict_pos = {}
    print("Reading individual output files")

    for x in range(num_tasks):
        file = os.path.join(eval_path, str(x) + '.txt')
        lines = open(file, 'r').readlines()[1:]
        for line in lines:
            line = line.strip()
            name = line.split('[')[0]
            label = line.split(']')[1].split(' ')[1]
            chunk_nb = line.split(']')[1].split(' ')[2]
            split_nb = line.split(']')[1].split(' ')[3]
            data = np.fromstring(line.split('[')[1].split(']')[0], dtype=np.float, sep=',')
            data = softmax(data)
            if not name in dict_feats:
                dict_feats[name] = []
                dict_label[name] = 0
                dict_pos[name] = []
            if chunk_nb + split_nb in dict_pos[name]:
                continue
            dict_feats[name].append(data)
            dict_pos[name].append(chunk_nb + split_nb)
            dict_label[name] = label
    print("Computing final results")

    input_lst = []
    print(len(dict_feats))
    for i, item in enumerate(dict_feats):
        input_lst.append([i, item, dict_feats[item], dict_label[item]])
    from multiprocessing import Pool
    p = Pool(64)
    ans = p.map(compute_video, input_lst)
    top1 = [x[1] for x in ans]
    top5 = [x[2] for x in ans]
    pred = [x[0] for x in ans]
    label = [x[3] for x in ans]
    final_top1 ,final_top5 = np.mean(top1), np.mean(top5)
    return final_top1*100 ,final_top5*100

def compute_video(lst):
    i, video_id, data, label = lst
    feat = [x for x in data]
    feat = np.mean(feat, axis=0)
    pred = np.argmax(feat)
    top1 = (int(pred) == int(label)) * 1.0
    top5 = (int(label) in np.argsort(-feat)[:5]) * 1.0
    return [pred, top1, top5, int(label)]
