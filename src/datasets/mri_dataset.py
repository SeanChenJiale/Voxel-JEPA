import os
import warnings
import numpy as np
import pandas as pd
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader, distributed
from src.datasets.utils.weighted_sampler import DistributedWeightedSampler
import torch
from torch.utils.data import Dataset
import nibabel as nib
import pandas as pd
import numpy as np

from logging import getLogger
_GLOBAL_SEED = 0
logger = getLogger()


def make_mridataset(
    data_paths,
    batch_size,
    num_slices=16,
    transform=None,
    rank=0,
    world_size=1,
    datasets_weights=None,
    collator=None,
    drop_last=True,
    num_workers=10,
    pin_mem=True,
    log_dir=None,
    gap = 2 # Step between slices to control spacing
):
    dataset = MRISliceDataset(
        data_paths=data_paths,
        # datasets_weights=datasets_weights,
        num_slices=num_slices, # num_frames # frames_per_clip
        # transform=transform,
        gap = gap
    )
    logger.info('MriDataset Successfully created')

    if datasets_weights is not None:
        dist_sampler = DistributedWeightedSampler(
            dataset.sample_weights,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
    else:
        dist_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )

    data_loader = DataLoader(
        dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=num_workers > 0
    )
    logger.info('MriDataset unsupervised data loader created successfully')
    # import pdb; pdb.set_trace()  # Put this where you want to pause execution
    return dataset, data_loader, dist_sampler

class MRISliceDataset(torch.utils.data.Dataset):
    def __init__(self, data_paths, datasets_weights=None, num_slices=48, transform=None, gap=2):
        self.data_paths = data_paths
        self.datasets_weights = datasets_weights
        self.num_slices = num_slices
        self.transform = transform
        self.gap = gap

        self.samples,self.axis, self.labels, self.num_samples_per_dataset = [], [], [], []
        for path in self.data_paths:
            df = pd.read_csv(path, header=None, delimiter=" ")
            self.samples += list(df.values[:, 0])
            self.axis += list(df.values[:, 1])
            self.labels += list(df.values[:, 2])
            self.num_samples_per_dataset.append(len(df))

        self.sample_weights = None
        if datasets_weights:
            self.sample_weights = []
            for w, n in zip(datasets_weights, self.num_samples_per_dataset):
                self.sample_weights += [w / n] * n

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        nii_path = self.samples[idx]
        label = self.labels[idx]
        
        # nib.load() loads the .nii.gz file as a Nibabel image object
        # .get_fdata() converts it to a NumPy ndarray of type float64 by default.
        volume = nib.load(nii_path).get_fdata()
        
        # Axis selection: 0 for sagittal, 1 for coronal, 2 for axial
        axis = int(self.axis[idx])  # Modified new lines
        # Transpose volume to make slicing axis last
        if axis == 0:
            volume = np.transpose(volume, (1, 2, 0))
        elif axis == 1:
            volume = np.transpose(volume, (0, 2, 1))
        else:
            volume = volume  # Already in axis=2
        # Axis selection successfully done

        target_size = 224 # max(volume.shape[0], volume.shape[1])
        pad_0 = ((target_size - volume.shape[0]) // 2, (target_size - volume.shape[0] + 1) // 2)
        pad_1 = ((target_size - volume.shape[1]) // 2, (target_size - volume.shape[1] + 1) // 2)
        volume = np.pad(volume, (pad_0, pad_1, (0, 0)), mode="constant")

        non_black_indices = [i for i in range(volume.shape[2]) if np.any(volume[:, :, i] > 0)]
        total_needed = self.num_slices * 3 * self.gap
        if len(non_black_indices) < total_needed:
            raise ValueError(f"Not enough slices in {nii_path}, required={total_needed}, found={len(non_black_indices)}")

        mid = len(non_black_indices) // 2
        center = non_black_indices[mid]
        selected = list(range(center - (total_needed // 2), center + (total_needed // 2), self.gap))[:self.num_slices * 3]

        slices = np.stack([volume[:, :, i] for i in selected], axis=0)
        slices = (slices - slices.min(axis=(1, 2), keepdims=True)) / (slices.max(axis=(1, 2), keepdims=True) - slices.min(axis=(1, 2), keepdims=True) + 1e-8)

        slices_tensor = torch.tensor(slices, dtype=torch.float32)
        rgb_frames = [torch.stack([slices_tensor[i + j] for j in range(3)], dim=-1) for i in range(0, self.num_slices * 3, 3)]
        buffer = torch.stack(rgb_frames, dim=0)  # [num_slices, H, W, 3]

        if self.transform:
            buffer = self.transform(buffer)
            
        # Rearrange from [T, H, W, C] → [C, T, H, W]
        buffer = buffer.permute(3, 0, 1, 2)  # [3, 16, 224, 224] #New line
        
        return [buffer], torch.tensor(label, dtype=torch.float32), torch.tensor(selected)