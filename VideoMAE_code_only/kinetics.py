import os
import numpy as np
from numpy.lib.function_base import disp
import torch
import decord
from PIL import Image
from torchvision import transforms
from random_erasing import RandomErasing
import warnings
from decord import VideoReader, cpu
from torch.utils.data import Dataset
import video_transforms as video_transforms 
import volume_transforms as volume_transforms
import nibabel as nib
from skimage.transform import resize
import pandas as pd
import ants


class VideoClsDataset(Dataset):
    """Load your own video classification dataset."""

    def __init__(self, anno_path, data_path, mode='train', clip_len=8,
                 frame_sample_rate=2, crop_size=224, short_side_size=256,
                 new_height=256, new_width=340, keep_aspect_ratio=True,
                 num_segment=1, num_crop=1, test_num_segment=10, test_num_crop=3,args=None):
        self.anno_path = anno_path
        self.data_path = data_path
        self.mode = mode
        self.clip_len = clip_len
        self.frame_sample_rate = frame_sample_rate
        self.crop_size = crop_size
        self.short_side_size = short_side_size
        self.new_height = new_height
        self.new_width = new_width
        self.keep_aspect_ratio = keep_aspect_ratio
        self.num_segment = num_segment
        self.test_num_segment = test_num_segment
        self.num_crop = num_crop
        self.test_num_crop = test_num_crop
        self.args = args
        self.aug = False
        self.rand_erase = False
        if self.mode in ['train']:
            self.aug = True
            if self.args.reprob > 0:
                self.rand_erase = True
        if VideoReader is None:
            raise ImportError("Unable to import `decord` which is required to read videos.")

        self.data_transform = video_transforms.Compose([
            video_transforms.Resize(self.short_side_size, interpolation='bilinear'),
            volume_transforms.ClipToTensor(),
            video_transforms.Normalize(mean=[0.146689, 0.146689, 0.146689],
                                        std=[0.267249, 0.267249, 0.267249])
        ])
            # self.test_seg = []
            # self.test_dataset = []
            # self.test_label_array = []
            # for ck in range(self.test_num_segment):
            #     for cp in range(self.test_num_crop):
            #         for idx in range(len(self.label_array)):
            #             sample_label = self.label_array[idx]
            #             self.test_label_array.append(sample_label)
            #             self.test_dataset.append(self.dataset_samples[idx])
            #             self.test_seg.append((ck, cp))

        self.strategy = 'AD'  # 'consecutive', 'skip_1', 'AD'
        self.frames_per_volume = 16  # number of slices per volume
        self.frame_step = 4  # temporal sampling rate
        self.num_clips = 1
        self.filter_short_volumes = False  # whether to filter out volumes with insufficient slices

        if self.mode == 'ood':
            samples, labels, axis= [], [], []
            data = pd.read_csv(self.anno_path, header=None, delimiter=" ") #-
            samples += list(data.values[:, 0])
            axis += list(data.values[:, 1])
            labels += list(data.values[:, 2])

            self.samples = samples
            self.labels = labels
            self.axis = axis


        else:
            self.samples = []
            self.labels = []
            self.axis = []
            self.indices = []
            data = pd.read_csv(self.anno_path)
            # Extract columns
            self.samples += list(data['filename'])
            self.axis += list(data['axis'])
            self.labels += list(data['label'])
            self.indices += [list(map(int, idx.split(','))) for idx in data['indices']]

            print(f"Dataset initialized with {len(self.samples)} samples for mode: {self.mode}")


    def __getitem__(self, idx):

        if self.mode == 'train':

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
            volume = split_into_clips(volume)[0].astype(np.uint8)
            # buffer = self._aug_frame(volume, self.args)
            buffer = self.data_transform(volume)  # use the same data augmentation as validation

            return buffer, label, idx, {}

        elif self.mode == 'validation':
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
            volume = split_into_clips(volume)[0].astype(np.uint8)
            buffer = self.data_transform(volume)
            return buffer, label, sample.split(".")[0]
        
        elif self.mode == 'test':
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
            volume = split_into_clips(volume)[0].astype(np.uint8)
            buffer = self.data_transform(volume)
            return buffer, label, sample.split(".")[0]
        
        elif self.mode == 'ood':
            sample = self.samples[idx]
            loaded_volume = False
            
            while not loaded_volume:
                try:
                    volume, slice_indices = self.load_volume_ood(idx)
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
            volume = split_into_clips(volume)[0].astype(np.uint8)
            buffer = self.data_transform(volume)
            return buffer, label, sample.split(".")[0]
        

    def __len__(self):
        # if self.mode != 'test':
        #     return len(self.dataset_samples)
        # else:
        #     return len(self.test_dataset)
        return len(self.samples)
        

    def load_volume_ood(self, idx):
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
        elif self.strategy == 'AD2':
            if axis == 0: # coronal
                # Focus on hippocampus (middle-posterior) and parietal regions
                # Move sampling towards posterior (larger indices)
                start = max(center - 12, first_non_black)  # Start slightly anterior to center
                end = start + 36  # Take 36 consecutive slices (more hippocampus coverage)
                indices_posterior = np.arange(start, end)
                
                # Add some anterior slices with sparser sampling for context
                anterior_end = start
                anterior_start = max(anterior_end - 24, first_non_black)
                indices_anterior = np.arange(anterior_start, anterior_end, 2)  # Every 2nd slice
                
                # Concatenate: anterior context + dense hippocampus region
                indices = np.concatenate((indices_anterior, indices_posterior))
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
        try:
            slices = filtered_data[indices]  # [T, H, W]
        except IndexError as e:
            # print(f"\n[ERROR] IndexError in volume: {sample}")
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
        return buffer, indices # buffer is T H W C


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


def spatial_sampling(
    frames,
    spatial_idx=-1,
    min_scale=256,
    max_scale=320,
    crop_size=224,
    random_horizontal_flip=True,
    inverse_uniform_sampling=False,
    aspect_ratio=None,
    scale=None,
    motion_shift=False,
):
    """
    Perform spatial sampling on the given video frames. If spatial_idx is
    -1, perform random scale, random crop, and random flip on the given
    frames. If spatial_idx is 0, 1, or 2, perform spatial uniform sampling
    with the given spatial_idx.
    Args:
        frames (tensor): frames of images sampled from the video. The
            dimension is `num frames` x `height` x `width` x `channel`.
        spatial_idx (int): if -1, perform random spatial sampling. If 0, 1,
            or 2, perform left, center, right crop if width is larger than
            height, and perform top, center, buttom crop if height is larger
            than width.
        min_scale (int): the minimal size of scaling.
        max_scale (int): the maximal size of scaling.
        crop_size (int): the size of height and width used to crop the
            frames.
        inverse_uniform_sampling (bool): if True, sample uniformly in
            [1 / max_scale, 1 / min_scale] and take a reciprocal to get the
            scale. If False, take a uniform sample from [min_scale,
            max_scale].
        aspect_ratio (list): Aspect ratio range for resizing.
        scale (list): Scale range for resizing.
        motion_shift (bool): Whether to apply motion shift for resizing.
    Returns:
        frames (tensor): spatially sampled frames.
    """
    assert spatial_idx in [-1, 0, 1, 2]
    if spatial_idx == -1:
        if aspect_ratio is None and scale is None:
            frames, _ = video_transforms.random_short_side_scale_jitter(
                images=frames,
                min_size=min_scale,
                max_size=max_scale,
                inverse_uniform_sampling=inverse_uniform_sampling,
            )
            frames, _ = video_transforms.random_crop(frames, crop_size)
        else:
            transform_func = (
                video_transforms.random_resized_crop_with_shift
                if motion_shift
                else video_transforms.random_resized_crop
            )
            frames = transform_func(
                images=frames,
                target_height=crop_size,
                target_width=crop_size,
                scale=scale,
                ratio=aspect_ratio,
            )
        if random_horizontal_flip:
            frames, _ = video_transforms.horizontal_flip(0.5, frames)
    else:
        # The testing is deterministic and no jitter should be performed.
        # min_scale, max_scale, and crop_size are expect to be the same.
        assert len({min_scale, max_scale, crop_size}) == 1
        frames, _ = video_transforms.random_short_side_scale_jitter(
            frames, min_scale, max_scale
        )
        frames, _ = video_transforms.uniform_crop(frames, crop_size, spatial_idx)
    return frames


def tensor_normalize(tensor, mean, std):
    """
    Normalize a given tensor by subtracting the mean and dividing the std.
    Args:
        tensor (tensor): tensor to normalize.
        mean (tensor or list): mean value to subtract.
        std (tensor or list): std to divide.
    """
    if tensor.dtype == torch.uint8:
        tensor = tensor.float()
        tensor = tensor / 255.0
    if type(mean) == list:
        mean = torch.tensor(mean)
    if type(std) == list:
        std = torch.tensor(std)
    tensor = tensor - mean
    tensor = tensor / std
    return tensor


# class VideoMAE(torch.utils.data.Dataset):
#     """Load your own video classification dataset.
#     Parameters
#     ----------
#     root : str, required.
#         Path to the root folder storing the dataset.
#     setting : str, required.
#         A text file describing the dataset, each line per video sample.
#         There are three items in each line: (1) video path; (2) video length and (3) video label.
#     train : bool, default True.
#         Whether to load the training or validation set.
#     test_mode : bool, default False.
#         Whether to perform evaluation on the test set.
#         Usually there is three-crop or ten-crop evaluation strategy involved.
#     name_pattern : str, default None.
#         The naming pattern of the decoded video frames.
#         For example, img_00012.jpg.
#     video_ext : str, default 'mp4'.
#         If video_loader is set to True, please specify the video format accordinly.
#     is_color : bool, default True.
#         Whether the loaded image is color or grayscale.
#     modality : str, default 'rgb'.
#         Input modalities, we support only rgb video frames for now.
#         Will add support for rgb difference image and optical flow image later.
#     num_segments : int, default 1.
#         Number of segments to evenly divide the video into clips.
#         A useful technique to obtain global video-level information.
#         Limin Wang, etal, Temporal Segment Networks: Towards Good Practices for Deep Action Recognition, ECCV 2016.
#     num_crop : int, default 1.
#         Number of crops for each image. default is 1.
#         Common choices are three crops and ten crops during evaluation.
#     new_length : int, default 1.
#         The length of input video clip. Default is a single image, but it can be multiple video frames.
#         For example, new_length=16 means we will extract a video clip of consecutive 16 frames.
#     new_step : int, default 1.
#         Temporal sampling rate. For example, new_step=1 means we will extract a video clip of consecutive frames.
#         new_step=2 means we will extract a video clip of every other frame.
#     temporal_jitter : bool, default False.
#         Whether to temporally jitter if new_step > 1.
#     video_loader : bool, default False.
#         Whether to use video loader to load data.
#     use_decord : bool, default True.
#         Whether to use Decord video loader to load data. Otherwise use mmcv video loader.
#     transform : function, default None.
#         A function that takes data and label and transforms them.
#     data_aug : str, default 'v1'.
#         Different types of data augmentation auto. Supports v1, v2, v3 and v4.
#     lazy_init : bool, default False.
#         If set to True, build a dataset instance without loading any dataset.
#     """
#     def __init__(self,
#                  root,
#                  setting,
#                  train=True,
#                  test_mode=False,
#                  name_pattern='img_%05d.jpg',
#                  video_ext='mp4',
#                  is_color=True,
#                  modality='rgb',
#                  num_segments=1,
#                  num_crop=1,
#                  new_length=1,
#                  new_step=1,
#                  transform=None,
#                  temporal_jitter=False,
#                  video_loader=False,
#                  use_decord=False,
#                  lazy_init=False):

#         super(VideoMAE, self).__init__()
#         self.root = root
#         self.setting = setting
#         self.train = train
#         self.test_mode = test_mode
#         self.is_color = is_color
#         self.modality = modality
#         self.num_segments = num_segments
#         self.num_crop = num_crop
#         self.new_length = new_length
#         self.new_step = new_step
#         self.skip_length = self.new_length * self.new_step
#         self.temporal_jitter = temporal_jitter
#         self.name_pattern = name_pattern
#         self.video_loader = video_loader
#         self.video_ext = video_ext
#         self.use_decord = use_decord
#         self.transform = transform
#         self.lazy_init = lazy_init


#         if not self.lazy_init:
#             self.clips = self._make_dataset(root, setting)
#             if len(self.clips) == 0:
#                 raise(RuntimeError("Found 0 video clips in subfolders of: " + root + "\n"
#                                    "Check your data directory (opt.data-dir)."))

#     def __getitem__(self, index):

#         directory, target = self.clips[index]
#         if self.video_loader:
#             if '.' in directory.split('/')[-1]:
#                 # data in the "setting" file already have extension, e.g., demo.mp4
#                 video_name = directory
#             else:
#                 # data in the "setting" file do not have extension, e.g., demo
#                 # So we need to provide extension (i.e., .mp4) to complete the file name.
#                 video_name = '{}.{}'.format(directory, self.video_ext)

#             decord_vr = decord.VideoReader(video_name, num_threads=1)
#             duration = len(decord_vr)

#         segment_indices, skip_offsets = self._sample_train_indices(duration)

#         images = self._video_TSN_decord_batch_loader(directory, decord_vr, duration, segment_indices, skip_offsets)

#         process_data, mask = self.transform((images, None)) # T*C,H,W
#         process_data = process_data.view((self.new_length, 3) + process_data.size()[-2:]).transpose(0,1)  # T*C,H,W -> T,C,H,W -> C,T,H,W
        
#         return (process_data, mask)

#     def __len__(self):
#         return len(self.clips)

#     def _make_dataset(self, directory, setting):
#         if not os.path.exists(setting):
#             raise(RuntimeError("Setting file %s doesn't exist. Check opt.train-list and opt.val-list. " % (setting)))
#         clips = []
#         with open(setting) as split_f:
#             data = split_f.readlines()
#             for line in data:
#                 line_info = line.split(' ')
#                 # line format: video_path, video_duration, video_label
#                 if len(line_info) < 2:
#                     raise(RuntimeError('Video input format is not correct, missing one or more element. %s' % line))
#                 clip_path = os.path.join(line_info[0])
#                 target = int(line_info[1])
#                 item = (clip_path, target)
#                 clips.append(item)
#         return clips

#     def _sample_train_indices(self, num_frames):
#         average_duration = (num_frames - self.skip_length + 1) // self.num_segments
#         if average_duration > 0:
#             offsets = np.multiply(list(range(self.num_segments)),
#                                   average_duration)
#             offsets = offsets + np.random.randint(average_duration,
#                                                   size=self.num_segments)
#         elif num_frames > max(self.num_segments, self.skip_length):
#             offsets = np.sort(np.random.randint(
#                 num_frames - self.skip_length + 1,
#                 size=self.num_segments))
#         else:
#             offsets = np.zeros((self.num_segments,))

#         if self.temporal_jitter:
#             skip_offsets = np.random.randint(
#                 self.new_step, size=self.skip_length // self.new_step)
#         else:
#             skip_offsets = np.zeros(
#                 self.skip_length // self.new_step, dtype=int)
#         return offsets + 1, skip_offsets


#     def _video_TSN_decord_batch_loader(self, directory, video_reader, duration, indices, skip_offsets):
#         sampled_list = []
#         frame_id_list = []
#         for seg_ind in indices:
#             offset = int(seg_ind)
#             for i, _ in enumerate(range(0, self.skip_length, self.new_step)):
#                 if offset + skip_offsets[i] <= duration:
#                     frame_id = offset + skip_offsets[i] - 1
#                 else:
#                     frame_id = offset - 1
#                 frame_id_list.append(frame_id)
#                 if offset + self.new_step < duration:
#                     offset += self.new_step
#         try:
#             video_data = video_reader.get_batch(frame_id_list).asnumpy()
#             sampled_list = [Image.fromarray(video_data[vid, :, :, :]).convert('RGB') for vid, _ in enumerate(frame_id_list)]
#         except:
#             raise RuntimeError('Error occured in reading frames {} from video {} of duration {}.'.format(frame_id_list, directory, duration))
#         return sampled_list


class VideoMAE(torch.utils.data.Dataset):
    """Load your own video classification dataset.
    Parameters
    ----------
    root : str, required.
        Path to the root folder storing the dataset.
    setting : str, required.
        A text file describing the dataset, each line per video sample.
        There are three items in each line: (1) video path; (2) video length and (3) video label.
    train : bool, default True.
        Whether to load the training or validation set.
    test_mode : bool, default False.
        Whether to perform evaluation on the test set.
        Usually there is three-crop or ten-crop evaluation strategy involved.
    name_pattern : str, default None.
        The naming pattern of the decoded video frames.
        For example, img_00012.jpg.
    video_ext : str, default 'mp4'.
        If video_loader is set to True, please specify the video format accordinly.
    is_color : bool, default True.
        Whether the loaded image is color or grayscale.
    modality : str, default 'rgb'.
        Input modalities, we support only rgb video frames for now.
        Will add support for rgb difference image and optical flow image later.
    num_segments : int, default 1.
        Number of segments to evenly divide the video into clips.
        A useful technique to obtain global video-level information.
        Limin Wang, etal, Temporal Segment Networks: Towards Good Practices for Deep Action Recognition, ECCV 2016.
    num_crop : int, default 1.
        Number of crops for each image. default is 1.
        Common choices are three crops and ten crops during evaluation.
    new_length : int, default 1.
        The length of input video clip. Default is a single image, but it can be multiple video frames.
        For example, new_length=16 means we will extract a video clip of consecutive 16 frames.
    new_step : int, default 1.
        Temporal sampling rate. For example, new_step=1 means we will extract a video clip of consecutive frames.
        new_step=2 means we will extract a video clip of every other frame.
    temporal_jitter : bool, default False.
        Whether to temporally jitter if new_step > 1.
    video_loader : bool, default False.
        Whether to use video loader to load data.
    use_decord : bool, default True.
        Whether to use Decord video loader to load data. Otherwise use mmcv video loader.
    transform : function, default None.
        A function that takes data and label and transforms them.
    data_aug : str, default 'v1'.
        Different types of data augmentation auto. Supports v1, v2, v3 and v4.
    lazy_init : bool, default False.
        If set to True, build a dataset instance without loading any dataset.
    """
    def __init__(self,
                 root,
                 setting,
                 use_cov_loss=False,
                 train=True,
                 test_mode=False,
                 name_pattern='img_%05d.jpg',
                 video_ext='mp4',
                 is_color=True,
                 modality='rgb',
                 num_segments=1,
                 num_crop=1,
                 new_length=1,
                 new_step=1,
                 transform=None,
                 temporal_jitter=False,
                 video_loader=False,
                 use_decord=False,
                 lazy_init=False):

        super(VideoMAE, self).__init__()
        self.root = root
        self.setting = setting
        self.train = train
        self.test_mode = test_mode
        self.is_color = is_color
        self.modality = modality
        self.num_segments = num_segments
        self.num_crop = num_crop
        self.new_length = new_length
        self.new_step = new_step
        self.skip_length = self.new_length * self.new_step
        self.temporal_jitter = temporal_jitter
        self.name_pattern = name_pattern
        self.video_loader = video_loader
        self.video_ext = video_ext
        self.use_decord = use_decord
        self.transform = transform
        self.lazy_init = lazy_init
        self.strategy = 'AD'  # 'consecutive', 'skip_1', 'AD'
        self.frames_per_volume = 16  # number of slices per volume
        self.frame_step = 4  # temporal sampling rate
        self.num_clips = 1
        self.filter_short_volumes = False  # whether to filter out volumes with insufficient slices
        self.use_cov_loss = use_cov_loss

        self.samples = []
        self.labels = []
        self.axis = []
        self.indices = []
        data = pd.read_csv(self.setting)
        # Extract columns
        self.samples += list(data['filename'])
        self.axis += list(data['axis'])
        self.labels += list(data['label'])
        self.indices += [list(map(int, idx.split(','))) for idx in data['indices']]

        print(f"Dataset initialized with {len(self.samples)} samples")

        # samples, labels, axis= [], [], []
        # self.num_samples_per_dataset = []

        # data = pd.read_csv(self.setting, header=None, delimiter=" ") #-
        # samples += list(data.values[:, 0])
        # axis += list(data.values[:, 1])
        # labels += list(data.values[:, 2])
        # num_samples = len(data)
        # self.num_samples_per_dataset.append(num_samples)

        # self.samples = samples
        # self.labels = labels
        # self.axis = axis


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

        axis = self.axis[idx]
        indices = self.indices[idx]
        volume = split_into_clips(volume)[0]
        images = [Image.fromarray(volume[t].astype(np.uint8)) for t in range(volume.shape[0])]
        process_data, mask = self.transform((images, None)) # T*C,H,W
        process_data = process_data.view((self.frames_per_volume, 3) + process_data.size()[-2:]).transpose(0,1)  # T*C,H,W -> T,C,H,W -> C,T,H,W
        
        if self.use_cov_loss:
            return (process_data, mask, torch.tensor(axis, dtype=torch.int), torch.tensor(indices, dtype=torch.int))
        else:
            return (process_data, mask)


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


    # def load_volume(self, idx):
    #     path = self.samples[idx]
    #     axis = self.axis[idx]

    #     # Load volume
    #     nii = nib.load(path)
    #     data = nii.get_fdata()
    #     filtered_data = data

    #     axis = int(axis) # Modified new lines

    #     # Transpose volume to make slicing axis last
    #     if axis == 0: # coronal
    #         filtered_data = np.transpose(filtered_data, (1, 2, 0))
    #     elif axis == 1: #axial
    #         filtered_data = np.transpose(filtered_data, (2, 0, 1))
    #     # 2 = sagittal, no change needed

    #     # Validate enough slices
    #     if filtered_data.shape[0] < self.frames_per_volume:
    #         if self.filter_short_volumes:
    #             return torch.zeros(1), 0, None
    #         else:
    #             # Pad last slice
    #             pad_count = self.frames_per_volume - filtered_data.shape[0]
    #             pad_slices = np.repeat(filtered_data[-1][np.newaxis, :, :], pad_count, axis=0)
    #             filtered_data = np.concatenate([filtered_data, pad_slices], axis=0)

    #     # Sample clip
    #     clip_len = self.frames_per_volume * self.frame_step

    #     # Find first and last non-black slice
    #     slice_sums = filtered_data.reshape(filtered_data.shape[0], -1).sum(axis=1)
    #     non_black_indices = np.where(slice_sums != 0)[0]
    #     num_slices = 48
    #     first_non_black = non_black_indices[0]
    #     last_non_black = non_black_indices[-1]
    #     center = (first_non_black + last_non_black) // 2
    #     available_slices = last_non_black - first_non_black + 1

    #     if self.strategy == 'consecutive':
    #         # Center 48 slices around the middle of the non-black region
    #         half = num_slices // 2
    #         start = max(center - half, 0)
    #         end = start + num_slices

    #         # Adjust if end goes beyond available slices
    #         if end > filtered_data.shape[0]:
    #             end = filtered_data.shape[0]
    #             start = max(end - num_slices, 0)

    #         indices = np.arange(start, end)
    #     elif self.strategy == 'skip_1':
    #         half = num_slices // 2
    #         start = max(center - half*2, 0)
    #         end = start + num_slices*2
    #         if end > filtered_data.shape[0]:
    #             end = filtered_data.shape[0]
    #             start = max(end - num_slices*2, 0)
    #         indices = np.arange(start, end, 2)
    #     elif self.strategy == 'AD':
    #         if axis == 0: # coronal
                
    #             start = max (center - 12,0)
    #             end = start + 24
    #             indices = np.arange(start, end)
    #             end = start
    #             start = max(end - 48,first_non_black)
    #             indices = np.concatenate(( np.arange(start, end,2), indices))
    #         elif axis == 1 or axis == 2: #axial or sagittal
    #             quartile = available_slices // 4
    #             start = first_non_black + quartile
    #             end = first_non_black + 3*quartile
    #             if end - start < 48:
    #                 start = max(center - 24, 0)
    #                 end = start + 48
    #             num_slices_to_work_with = end - start
    #             step = num_slices_to_work_with // 48
    #             indices = np.arange(start, end, step) #take the center 48 in the list
    #             if len(indices) > 48:
    #                 center = len(indices) // 2
    #                 half = 48 // 2
    #                 indices = indices[center - half : center + half]
    #             # find  
    #     elif self.strategy == 'AD2':
    #         if axis == 0: # coronal
    #             # Focus on hippocampus (middle-posterior) and parietal regions
    #             # Move sampling towards posterior (larger indices)
    #             start = max(center - 12, first_non_black)  # Start slightly anterior to center
    #             end = start + 36  # Take 36 consecutive slices (more hippocampus coverage)
    #             indices_posterior = np.arange(start, end)
                
    #             # Add some anterior slices with sparser sampling for context
    #             anterior_end = start
    #             anterior_start = max(anterior_end - 24, first_non_black)
    #             indices_anterior = np.arange(anterior_start, anterior_end, 2)  # Every 2nd slice
                
    #             # Concatenate: anterior context + dense hippocampus region
    #             indices = np.concatenate((indices_anterior, indices_posterior))
    #         elif axis == 1 or axis == 2: #axial or sagittal
    #             quartile = available_slices // 4
    #             start = first_non_black + quartile
    #             end = first_non_black + 3*quartile
    #             if end - start < 48:
    #                 start = max(center - 24, 0)
    #                 end = start + 48
    #             num_slices_to_work_with = end - start
    #             step = num_slices_to_work_with // 48
    #             indices = np.arange(start, end, step) #take the center 48 in the list
    #             if len(indices) > 48:
    #                 center = len(indices) // 2
    #                 half = 48 // 2
    #                 indices = indices[center - half : center + half]
    #     try:
    #         slices = filtered_data[indices]  # [T, H, W]
    #     except IndexError as e:
    #         # print(f"\n[ERROR] IndexError in volume: {sample}")
    #         print(f" - filtered_data.shape[0] = {filtered_data.shape[0]}")
    #         print(f" - indices = {indices}")
    #         raise e

    #     mins = slices.min(axis=(1, 2), keepdims=True)  # shape [num_slices, 1, 1]
    #     maxs = slices.max(axis=(1, 2), keepdims=True)  # shape [num_slices, 1, 1]

    #     # Normalize each slice to [0, 255]
    #     norm_slices = np.where(
    #         maxs > mins,
    #         255.0 * (slices - mins) / (maxs - mins),
    #         0
    #     )
    #     #resize each slice to 224,224 
    #     norm_slices = np.array([resize(slice, (224, 224), anti_aliasing=True) for slice in norm_slices])

    #     # # Round and convert to integer type
    #     # norm_slices = np.round(norm_slices).astype(np.uint8)
    #     #### new
    #     # Stack every 3 consecutive normalized slices into a 3-channel image change to slices if norm is not needed
    #     num_rgb_frames = slices.shape[0] // 3
    #     # rgb_frames = [np.stack([slices[i + j] for j in range(3)], axis=-1)
    #     #               for i in range(0, num_rgb_frames * 3, 3)]
    #     # After normalization
    #     rgb_frames = [np.stack([norm_slices[i + j] for j in range(3)], axis=-1)
    #                   for i in range(0, num_rgb_frames * 3, 3)]

    #     buffer = np.stack(rgb_frames, axis=0)  # [T, H, W, 3]
    #     # Package into list of clips like video dataset (even if num_clips=1)
    #     indices = [indices]
    #     return buffer, indices # buffer is T H W C

