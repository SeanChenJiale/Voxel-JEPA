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
import ants
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
        # Load data from CSV
        self.samples = []
        self.labels = []
        self.axis = []
        self.indices = []

        for data_path in self.data_paths:
            if data_path.endswith('.csv'):
                # Read the CSV file
                data = pd.read_csv(data_path)
                # Extract columns
                self.samples += list(data['filename'])
                self.axis += list(data['axis'])
                self.labels += list(data['label'])
                self.indices += [list(map(int, idx.split(','))) for idx in data['indices']]


        self.num_samples = len(self.samples)
        # volume, slice_indices = self.load_volume(1)



    def __len__(self):
        return len(self.samples)


    def load_volume(self, idx):
        path = self.samples[idx]
        axis = self.axis[idx]
        label = self.labels[idx]
        indices = self.indices[idx]
        # Load volume
        nii = ants.image_read(path)
        filtered_data = nii.numpy()
        #resize each slice to 224,224 
        norm_slices = np.array([resize(slice_c, (224, 224), anti_aliasing=True) for slice_c in filtered_data])

        # Stack every 3 consecutive normalized slices into a 3-channel image change to slices if norm is not needed
        num_rgb_frames = norm_slices.shape[0] // 3

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
                print(sample)
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

        if self.transform is not None:
            volume = [self.transform(clip) for clip in volume]
        return volume, label, slice_indices, sample , idx , axis

