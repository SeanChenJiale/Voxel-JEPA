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

_GLOBAL_SEED = 0
logger = getLogger()


def visualize_slices(slices_tensor, n_cols=4):
    """
    Visualize T slices from a tensor of shape [1, T, H, W].
    """
    # if isinstance(slices_tensor, torch.Tensor):
    #     slices_tensor = slices_tensor.squeeze(0)  # [T, H, W]

    T = slices_tensor.shape[0]
    n_rows = (T + n_cols - 1) // n_cols  # Compute number of rows for grid

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    axs = axs.flatten()

    for i in range(T):
        axs[i].imshow(slices_tensor[i], cmap='gray')
        axs[i].axis('off')
        axs[i].set_title(f"Slice {i}")

    # Hide any unused subplots
    for i in range(T, len(axs)):
        axs[i].axis('off')

    plt.tight_layout()
    plt.show()


def make_mridataset(
    data_paths,
    batch_size,
    frames_per_volume=8,
    frame_step=4,
    num_clips=1,
    random_clip_sampling=True,
    allow_clip_overlap=False,
    filter_short_volumes=False,
    filter_long_volumes=int(10**9),
    transform=None,
    shared_transform=None,
    rank=0,
    world_size=1,
    datasets_weights=None,
    collator=None,
    drop_last=True,
    num_workers=10,
    pin_mem=True,
    duration=None,
    log_dir=None,
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
        transform=transform

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

        # Load video paths and labels
        samples, labels = [], []
        self.num_samples_per_dataset = []
        for data_path in self.data_paths:
            if data_path[-4:] == '.csv':
                data = pd.read_csv(data_path, header=None, delimiter=" ") #-
                samples += list(data.values[:, 0])
                labels += list(data.values[:, 1])
                num_samples = len(data)
                self.num_samples_per_dataset.append(num_samples)

        self.samples = samples
        self.labels = labels
        # self.load_volume(0)


    def __len__(self):
        return len(self.samples)


    def load_volume(self, sample):
        # path = self.samples[idx]
        # label = self.labels[idx]

        # img = nib.load(path)
        # data = img.get_fdata()
        # current_ornt = io_orientation(img.affine)
        # target_ornt = axcodes2ornt(('R', 'A', 'S'))  # canonical orientation
        # transform = ornt_transform(current_ornt, target_ornt)
        # canonical_data = nib.orientations.apply_orientation(data, transform)
        # rotated_volume = np.rot90(canonical_data, k=1, axes=(1, 2))

        # Load volume
        nii = nib.load(sample)
        data = nii.get_fdata()

        # Get current orientation
        affine = nii.affine
        ornt = io_orientation(affine)

        # Get transformation to RAS
        ras_ornt = axcodes2ornt(('R', 'A', 'S'))  # standard RAS orientation
        transform = nib.orientations.ornt_transform(ornt, ras_ornt)

        # Apply reorientation
        reoriented_data = apply_orientation(data, transform)
        reoriented_data = np.rot90(reoriented_data, k=1, axes=(1,2))  # Rotate XY slices

        # reoriented_data is a 3D numpy array: [D, H, W]
        # Find non-blank slices (i.e., at least one non-zero voxel in the slice)
        non_blank_mask = np.any(reoriented_data != 0, axis=(1, 2))  # shape: (D,)

        # Filter the volume to keep only non-blank slices
        filtered_data = reoriented_data[non_blank_mask]

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


        #=========================++++++===============
        # max_start = filtered_data.shape[0] - (self.frames_per_volume - 1) * self.frame_step
        #
        # if max_start <= 0:
        #     start_idx = 0
        # else:
        #     start_idx = np.random.randint(0, max_start) if self.random_clip_sampling else 0
        #
        # indices = np.arange(start_idx, start_idx + clip_len, self.frame_step)
        #===============================================
        max_start = filtered_data.shape[0] - clip_len   #-
        start_idx = np.random.randint(0, max(1, max_start + 1)) if self.random_clip_sampling else 0  #-
        indices = np.arange(start_idx, start_idx + clip_len, self.frame_step) #-
        indices = indices[:self.frames_per_volume]

        # Select slices
        # slices = filtered_data[indices]  # [T, H, W]
        try:
            slices = filtered_data[indices]  # [T, H, W]
        except IndexError as e:
            print(f"\n[ERROR] IndexError in volume: {sample}")
            print(f" - filtered_data.shape[0] = {filtered_data.shape[0]}")
            print(f" - indices = {indices}")
            raise e

        # visualize_slices(slices)
        # pdb.set_trace()
        # if not os.path.exists(path):
        #     warnings.warn(f"File not found: {path}")
        #     return torch.zeros(1), label, None
        #
        # img = nib.load(path).get_fdata().astype(np.float32)
        #
        # # Transpose to [D, H, W] => [H, W, D]
        # if img.shape[0] != img.shape[1]:
        #     img = np.transpose(img, (1, 0, 2))  # to HWC if needed
        #
        # # Remove blank slices (sum < epsilon)
        # epsilon = 1e-3
        # slice_sums = np.sum(img, axis=(0, 1))
        # valid_indices = np.where(slice_sums > epsilon)[0]

        # Output as [1, T, H, W]
        # slices = torch.from_numpy(slices).unsqueeze(0).permute(0, 3, 1, 2)  # [1, T, H, W]
        # return slices, label, indices.tolist()

        # Final formatting to mimic video-style [T, H, W, C]
        # MRI is grayscale → add a dummy channel dimension (C=1)
        slices = slices[..., np.newaxis]  # [T, H, W, 1]

        # Package into list of clips like video dataset (even if num_clips=1)
        indices = [indices]
        return slices, indices

    def __getitem__(self, idx):
        sample = self.samples[idx]

        loaded_volume = False
        # while not loaded_volume:
        #     volume, slice_indices = self.load_volume(sample)
        #     loaded_volume = len(volume)>0
        while not loaded_volume:
            try:
                volume, slice_indices = self.load_volume(sample)
                loaded_volume = len(volume) > 0
            except Exception as e:
                print(f"\n[DATASET ERROR] Failed at index {idx}, sample: {sample}")
                print(f"Exception: {e}")
                loaded_volume = False
            if not loaded_volume:
                index = np.random.randint(self.__len__())
                sample = self.samples[index]

        label = self.labels[idx]

        # Parse video into frames & apply data augmentations
        if self.shared_transform is not None:
            volume = self.shared_transform(volume)

        if self.transform is not None:
            buffer = [self.transform(volume)]

        return buffer, label, slice_indices, sample , index



def make_videodataset(
    data_paths,
    batch_size,
    frames_per_clip=8,
    frame_step=4,
    num_clips=1,
    random_clip_sampling=True,
    allow_clip_overlap=False,
    filter_short_videos=False,
    filter_long_videos=int(10**9),
    transform=None,
    shared_transform=None,
    rank=0,
    world_size=1,
    datasets_weights=None,
    collator=None,
    drop_last=True,
    num_workers=10,
    pin_mem=True,
    duration=None,
    log_dir=None,
):
    dataset = VideoDataset(
        data_paths=data_paths,
        datasets_weights=datasets_weights,
        frames_per_clip=frames_per_clip,
        frame_step=frame_step,
        num_clips=num_clips,
        random_clip_sampling=random_clip_sampling,
        allow_clip_overlap=allow_clip_overlap,
        filter_short_videos=filter_short_videos,
        filter_long_videos=filter_long_videos,
        duration=duration,
        shared_transform=shared_transform,
        transform=transform)

    logger.info('VideoDataset dataset created')
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
    logger.info('VideoDataset unsupervised data loader created')

    return dataset, data_loader, dist_sampler


class VideoDataset(torch.utils.data.Dataset):
    """ Video classification dataset. """

    def __init__(
        self,
        data_paths,
        datasets_weights=None,
        frames_per_clip=16,
        frame_step=4,
        num_clips=1,
        transform=None,
        shared_transform=None,
        random_clip_sampling=True,
        allow_clip_overlap=False,
        filter_short_videos=False,
        filter_long_videos=int(10**9),
        duration=None,  # duration in seconds
    ):
        self.data_paths = data_paths
        self.datasets_weights = datasets_weights
        self.frames_per_clip = frames_per_clip
        self.frame_step = frame_step
        self.num_clips = num_clips
        self.transform = transform
        self.shared_transform = shared_transform
        self.random_clip_sampling = random_clip_sampling
        self.allow_clip_overlap = allow_clip_overlap
        self.filter_short_videos = filter_short_videos
        self.filter_long_videos = filter_long_videos
        self.duration = duration
        pdb.set_trace()

        if VideoReader is None:
            raise ImportError('Unable to import "decord" which is required to read videos.')

        # Load video paths and labels
        samples, labels = [], []
        self.num_samples_per_dataset = []
        for data_path in self.data_paths:

            if data_path[-4:] == '.csv':
                data = pd.read_csv(data_path, header=None, delimiter=",")  #+
                # data = pd.read_csv(data_path, header=None, delimiter=" ") #-
                samples += list(data.values[:, 0])
                labels += list(data.values[:, 1])
                num_samples = len(data)
                self.num_samples_per_dataset.append(num_samples)

            elif data_path[-4:] == '.npy':
                data = np.load(data_path, allow_pickle=True)
                data = list(map(lambda x: repr(x)[1:-1], data))
                samples += data
                labels += [0] * len(data)
                num_samples = len(data)
                self.num_samples_per_dataset.append(len(data))

        # [Optional] Weights for each sample to be used by downstream
        # weighted video sampler
        self.sample_weights = None
        if self.datasets_weights is not None:
            self.sample_weights = []
            for dw, ns in zip(self.datasets_weights, self.num_samples_per_dataset):
                self.sample_weights += [dw / ns] * ns

        self.samples = samples
        self.labels = labels

    def __getitem__(self, index):
        sample = self.samples[index]

        # Keep trying to load videos until you find a valid sample
        loaded_video = False
        while not loaded_video:
            buffer, clip_indices = self.loadvideo_decord(sample)  # [T H W 3]
            loaded_video = len(buffer) > 0
            if not loaded_video:
                index = np.random.randint(self.__len__())
                sample = self.samples[index]

        # Label/annotations for video
        label = self.labels[index]

        def split_into_clips(video):
            """ Split video into a list of clips """
            fpc = self.frames_per_clip
            nc = self.num_clips
            return [video[i*fpc:(i+1)*fpc] for i in range(nc)]

        # Parse video into frames & apply data augmentations
        if self.shared_transform is not None:
            buffer = self.shared_transform(buffer)
        buffer = split_into_clips(buffer)
        if self.transform is not None:
            buffer = [self.transform(clip) for clip in buffer]

        return buffer, label, clip_indices

    def loadvideo_decord(self, sample):
        """ Load video content using Decord """

        fname = sample
        if not os.path.exists(fname):
            warnings.warn(f'video path not found {fname=}')
            return [], None

        _fsize = os.path.getsize(fname)
        if _fsize < 1 * 1024:  # avoid hanging issue
            warnings.warn(f'video too short {fname=}')
            return [], None
        if _fsize > self.filter_long_videos:
            warnings.warn(f'skipping long video of size {_fsize=} (bytes)')
            return [], None

        try:
            vr = VideoReader(fname, num_threads=-1, ctx=cpu(0))
        except Exception:
            return [], None

        fpc = self.frames_per_clip
        fstp = self.frame_step
        if self.duration is not None:
            try:
                fps = vr.get_avg_fps()
                fstp = int(self.duration * fps / fpc)
            except Exception as e:
                warnings.warn(e)
        clip_len = int(fpc * fstp)

        if self.filter_short_videos and len(vr) < clip_len:
            warnings.warn(f'skipping video of length {len(vr)}')
            return [], None

        vr.seek(0)  # Go to start of video before sampling frames

        # Partition video into equal sized segments and sample each clip
        # from a different segment
        partition_len = len(vr) // self.num_clips

        all_indices, clip_indices = [], []
        for i in range(self.num_clips):

            if partition_len > clip_len:
                # If partition_len > clip len, then sample a random window of
                # clip_len frames within the segment
                end_indx = clip_len
                if self.random_clip_sampling:
                    end_indx = np.random.randint(clip_len, partition_len)
                start_indx = end_indx - clip_len
                indices = np.linspace(start_indx, end_indx, num=fpc)
                indices = np.clip(indices, start_indx, end_indx-1).astype(np.int64)
                # --
                indices = indices + i * partition_len
            else:
                # If partition overlap not allowed and partition_len < clip_len
                # then repeatedly append the last frame in the segment until
                # we reach the desired clip length
                if not self.allow_clip_overlap:
                    indices = np.linspace(0, partition_len, num=partition_len // fstp)
                    indices = np.concatenate((indices, np.ones(fpc - partition_len // fstp) * partition_len,))
                    indices = np.clip(indices, 0, partition_len-1).astype(np.int64)
                    # --
                    indices = indices + i * partition_len

                # If partition overlap is allowed and partition_len < clip_len
                # then start_indx of segment i+1 will lie within segment i
                else:
                    sample_len = min(clip_len, len(vr)) - 1
                    indices = np.linspace(0, sample_len, num=sample_len // fstp)
                    indices = np.concatenate((indices, np.ones(fpc - sample_len // fstp) * sample_len,))
                    indices = np.clip(indices, 0, sample_len-1).astype(np.int64)
                    # --
                    clip_step = 0
                    if len(vr) > clip_len:
                        clip_step = (len(vr) - clip_len) // (self.num_clips - 1)
                    indices = indices + i * clip_step

            clip_indices.append(indices)
            all_indices.extend(list(indices))

        buffer = vr.get_batch(all_indices).asnumpy()
        return buffer, clip_indices

    def __len__(self):
        return len(self.samples)
