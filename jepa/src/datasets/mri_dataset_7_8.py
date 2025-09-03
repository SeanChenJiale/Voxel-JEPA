# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import pathlib
import pdb
import warnings

from logging import getLogger

import numpy as np
import pandas as pd

from decord import VideoReader, cpu

import torch
import nibabel as nib
from nibabel.orientations import axcodes2ornt, ornt_transform, io_orientation, apply_orientation
import matplotlib.pyplot as plt
from src.datasets.utils.weighted_sampler import DistributedWeightedSampler
from skimage.transform import resize

_GLOBAL_SEED = 0
logger = getLogger()

def make_mridataset(
    data_paths,
    batch_size,
    frames_per_volume=8,
    frame_step=4,
    duration=None,
    num_clips=1,
    random_clip_sampling=True,
    allow_clip_overlap=False,
    filter_short_volumes=False,
    filter_long_volumes=int(10**9),
    shared_transform=None,
    transform=None,
    datasets_weights=None,
    collator=None,
    drop_last=True,
    num_workers=10,
    rank=0,
    world_size=1,
    pin_mem=True,
    log_dir=None,
    debug=False,
    strategy='consecutive'
        ):
    dataset = MRIDataset(
        data_paths=data_paths,
        datasets_weights=datasets_weights,
        frames_per_volume=frames_per_volume,
        frame_step=frame_step,
        num_clips=num_clips,
        random_clip_sampling=random_clip_sampling,
        allow_clip_overlap=allow_clip_overlap,
        filter_short_volumes=filter_short_volumes,
        filter_long_volumes=filter_long_volumes,
        duration=duration,
        shared_transform=shared_transform,
        transform=transform,
        strategy=strategy,

    )
    logger.info('MRIDataset dataset created')
    if datasets_weights is not None:
        dist_sampler = DistributedWeightedSampler(
            dataset.sample_weights,
            num_replicas=world_size,
            rank=rank,
            shuffle=True)
    else:
        dist_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=num_workers > 0)
    logger.info('MRIDataset unsupervised data loader created')
    return dataset, data_loader, dist_sampler


class MRIDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_paths,
        datasets_weights=None,
        frames_per_volume=16,
        frame_step=1,
        num_clips=1,
        transform=None,
        shared_transform=None,
        random_clip_sampling=True,
        allow_clip_overlap=False,
        filter_short_volumes=False,
        filter_long_volumes=int(1e9),
        duration=None,
        debug=False, # for debugging
        save_csv_path=None, # for debugging
        strategy='consecutive' # 'default' or 'slice_based'
    ):
        self.data_paths = data_paths
        self.datasets_weights = datasets_weights
        self.frames_per_volume = frames_per_volume
        self.frame_step = frame_step
        self.num_clips = num_clips
        self.transform = transform
        self.shared_transform = shared_transform
        self.random_clip_sampling = random_clip_sampling
        self.allow_clip_overlap = allow_clip_overlap
        self.filter_short_volumes = filter_short_volumes
        self.filter_long_volumes = filter_long_volumes
        self.duration = duration
        self.strategy = strategy
        # Load video paths and labels
        samples, labels, axis= [], [], []
        self.num_samples_per_dataset = []
        for data_path in self.data_paths:
            if data_path[-4:] == '.csv':
                data = pd.read_csv(data_path, header=None, delimiter=" ") #-
                samples += list(data.values[:, 0])
                axis += list(data.values[:, 1])
                labels += list(data.values[:, 2])
                num_samples = len(data)
                self.num_samples_per_dataset.append(num_samples)

        self.samples = samples
        self.labels = labels
        self.axis = axis
        # self.load_volume(0)


    def __len__(self):
        return len(self.samples)


    def load_volume(self, idx):
        path = self.samples[idx]
        axis = self.axis[idx]

        # Load volume
        nii = nib.load(path)
        data = nii.get_fdata()
        filtered_data = data

        axis = int(axis) # Modified new lines

        # Transpose volume to make slicing axis last
        if axis == 0: # coronal
            filtered_data = np.transpose(filtered_data, (1, 2, 0))
        elif axis == 1: #axial
            filtered_data = np.transpose(filtered_data, (2, 0, 1))
        # 2 = sagittal, no change needed

        # Validate enough slices
        if filtered_data.shape[0] < self.frames_per_volume:
            if self.filter_short_volumes:
                return torch.zeros(1), 0, None
            else:
                # Pad last slice
                pad_count = self.frames_per_volume - filtered_data.shape[0]
                pad_slices = np.repeat(filtered_data[-1][np.newaxis, :, :], pad_count, axis=0)
                filtered_data = np.concatenate([filtered_data, pad_slices], axis=0)

        # Sample clip
        clip_len = self.frames_per_volume * self.frame_step

        # Find first and last non-black slice
        slice_sums = filtered_data.reshape(filtered_data.shape[0], -1).sum(axis=1)
        non_black_indices = np.where(slice_sums != 0)[0]
        num_slices = 48
        first_non_black = non_black_indices[0]
        last_non_black = non_black_indices[-1]
        center = (first_non_black + last_non_black) // 2
        available_slices = last_non_black - first_non_black + 1

        if self.strategy == 'consecutive':
            # Center 48 slices around the middle of the non-black region
            half = num_slices // 2
            start = max(center - half, 0)
            end = start + num_slices

            # Adjust if end goes beyond available slices
            if end > filtered_data.shape[0]:
                end = filtered_data.shape[0]
                start = max(end - num_slices, 0)

            indices = np.arange(start, end)
        elif self.strategy == 'skip_1':
            half = num_slices // 2
            start = max(center - half*2, 0)
            end = start + num_slices*2
            if end > filtered_data.shape[0]:
                end = filtered_data.shape[0]
                start = max(end - num_slices*2, 0)
            indices = np.arange(start, end, 2)
        elif self.strategy == 'AD':
            if axis == 0: # coronal
                
                start = max (center - 12,0)
                end = start + 24
                indices = np.arange(start, end)
                end = start
                start = max(end - 48,first_non_black)
                indices = np.concatenate(( np.arange(start, end,2), indices))
            elif axis == 1 or axis == 2: #axial or sagittal
                quartile = available_slices // 4
                start = first_non_black + quartile
                end = first_non_black + 3*quartile
                if end - start < 48:
                    start = max(center - 24, 0)
                    end = start + 48
                num_slices_to_work_with = end - start
                step = num_slices_to_work_with // 48
                indices = np.arange(start, end, step) #take the center 48 in the list
                if len(indices) > 48:
                    center = len(indices) // 2
                    half = 48 // 2
                    indices = indices[center - half : center + half]
                # find  

        try:
            slices = filtered_data[indices]  # [T, H, W]
        except IndexError as e:
            print(f"\n[ERROR] IndexError in volume: {sample}")
            print(f" - filtered_data.shape[0] = {filtered_data.shape[0]}")
            print(f" - indices = {indices}")
            raise e

        mins = slices.min(axis=(1, 2), keepdims=True)  # shape [num_slices, 1, 1]
        maxs = slices.max(axis=(1, 2), keepdims=True)  # shape [num_slices, 1, 1]

        # Normalize each slice to [0, 255]
        norm_slices = np.where(
            maxs > mins,
            255.0 * (slices - mins) / (maxs - mins),
            0
        )
        #resize each slice to 224,224 
        norm_slices = np.array([resize(slice, (224, 224), anti_aliasing=True) for slice in norm_slices])

        # # Round and convert to integer type
        # norm_slices = np.round(norm_slices).astype(np.uint8)
        #### new
        # Stack every 3 consecutive normalized slices into a 3-channel image change to slices if norm is not needed
        num_rgb_frames = slices.shape[0] // 3
        # rgb_frames = [np.stack([slices[i + j] for j in range(3)], axis=-1)
        #               for i in range(0, num_rgb_frames * 3, 3)]
        # After normalization
        rgb_frames = [np.stack([norm_slices[i + j] for j in range(3)], axis=-1)
                      for i in range(0, num_rgb_frames * 3, 3)]

        buffer = np.stack(rgb_frames, axis=0)  # [T, H, W, 3]
        # Package into list of clips like video dataset (even if num_clips=1)
        indices = [indices]
        return buffer, indices

    def __getitem__(self, idx):
        sample = self.samples[idx]

        loaded_volume = False

        while not loaded_volume:
            try:
                volume, slice_indices = self.load_volume(idx)
                # loaded_volume = len(volume) > 0
                loaded_volume = volume[0].shape[1] > 0  # T > 0
            except Exception as e:
                print(f"\n[DATASET ERROR] Failed at index {idx}, sample: {sample}")
                print(f"Exception: {e}")
                loaded_volume = False
            if not loaded_volume:
                index = np.random.randint(self.__len__())
                sample = self.samples[idx]
        def split_into_clips(volume):
            """ Split volume into a list of clips """
            fpc = self.frames_per_volume
            nc = self.num_clips
            return [volume[i*fpc:(i+1)*fpc] for i in range(nc)]
        label = self.labels[idx]
        axis = self.axis[idx]

        # Parse video into frames & apply data augmentations
        if self.shared_transform is not None:
            volume = self.shared_transform(volume)
        volume = split_into_clips(volume)
        # import os
        # from torchvision.utils import save_image
        # debug_dir = "/media/backup_16TB/sean/VJEPA/a6000_output/vit_base/K400/video_classification_frozen"
        # vid_or_mri_str = "debug_mri"
        # debug_dir = os.path.join(debug_dir, vid_or_mri_str,"iteration_epoch_clip")    
        # torch.save(volume[0], os.path.join(debug_dir, "debug_dataset_before_transform_tensor.pt"))
        if self.transform is not None:
            volume = [self.transform(clip) for clip in volume]
        # torch.save(volume[0], os.path.join(debug_dir, "debug_dataset_after_transform_tensor.pt"))
        # breakpoint()
        return volume, label, slice_indices, sample , idx , axis

