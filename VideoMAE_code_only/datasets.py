import os
from torchvision import transforms
from transforms import *
from masking_generator import TubeMaskingGenerator, HalfMaskingGenerator
from kinetics import VideoClsDataset, VideoMAE
from ssv2 import SSVideoClsDataset


class DataAugmentationForVideoMAE(object):
    def __init__(self, args):
        self.input_mean = [0.146689, 0.146689, 0.146689] #[0.485, 0.456, 0.406]  # IMAGENET_DEFAULT_MEAN
        self.input_std = [0.267249, 0.267249, 0.267249] #[0.229, 0.224, 0.225]  # IMAGENET_DEFAULT_STD
        normalize = GroupNormalize(self.input_mean, self.input_std)
        self.train_augmentation = GroupMultiScaleCrop(args.input_size, [1, .875, .75, .66])
        self.transform = transforms.Compose([                            
            self.train_augmentation,
            Stack(roll=False),
            ToTorchFormatTensor(div=True),
            normalize,
        ])
        if args.mask_type == 'tube':
            self.masked_position_generator = TubeMaskingGenerator(
                args.window_size, args.mask_ratio
            )
        elif args.mask_type == 'half':
            self.masked_position_generator = HalfMaskingGenerator(
                args.window_size, args.mask_ratio
            )

    def __call__(self, images):
        process_data, _ = self.transform(images)
        return process_data, self.masked_position_generator()

    def __repr__(self):
        repr = "(DataAugmentationForVideoMAE,\n"
        repr += "  transform = %s,\n" % str(self.transform)
        repr += "  Masked position generator = %s,\n" % str(self.masked_position_generator)
        repr += ")"
        return repr


def build_pretraining_dataset(args):
    transform = DataAugmentationForVideoMAE(args)
    dataset = VideoMAE(
        root=None,
        setting=args.data_path,
        use_cov_loss=args.use_cov_loss,
        video_ext='mp4',
        is_color=True,
        modality='rgb',
        new_length=args.num_frames,
        new_step=args.sampling_rate,
        transform=transform,
        temporal_jitter=False,
        video_loader=True,
        use_decord=True,
        lazy_init=False)
    print("Data Aug = %s" % str(transform))
    return dataset


def build_dataset(is_train, test_mode, args):
    def get_anno_path(base_path, filename, train_pct, is_train):
        if not is_train or train_pct == 100:
            path = os.path.join(base_path, filename)
            if not os.path.exists(path) and "IXI" in filename:
                # Fallback for IXI files if not in base_path
                fallback_path = os.path.join('/home/tianze/DATA_2T/MRI/IXI2_T1/meta', filename)
                if os.path.exists(fallback_path):
                    return fallback_path
            if not os.path.exists(path):
                print(f"Warning: Annotation file not found at {path}")
            return path
        
        # Determine the suffix based on percentage
        suffix = f"_{train_pct}"
        
        # Handle different naming conventions for splits
        if filename.endswith('.csv'):
            name_part = filename[:-4]
            if "IXI" in filename:
                # e.g., train_age_IXI_allscan.csv -> train_age_IXI_allscantrain_split_25.csv
                path = os.path.join(base_path, name_part + "train_split" + suffix + ".csv")
            elif "allscans_train" in filename:
                # e.g., NIFD_allscans_train.csv -> NIFD_allscans_train_split_25.csv
                path = os.path.join(base_path, name_part + "_split" + suffix + ".csv")
            else:
                # e.g., AD_MCI_CN_allscantrain_split.csv -> AD_MCI_CN_allscantrain_split_25.csv
                path = os.path.join(base_path, name_part + suffix + ".csv")
            
            if not os.path.exists(path):
                print(f"Warning: Annotation file not found at {path}")
            return path
        return os.path.join(base_path, filename)

    if args.data_set == 'AD_CN':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = get_anno_path(args.data_path, 'AD_CN_allscantrain_split.csv', args.train_pct, is_train)
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'AD_CN_allscantest_split.csv') 
        else:  
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'AD_CN_allscanval_split.csv') 

        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 2

    elif args.data_set == 'AD_CN_ood':

        mode = 'ood'
        anno_path = os.path.join(args.data_path, 'OASIS3_final.csv')

        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 2

    elif args.data_set == 'MCI_CN':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = get_anno_path(args.data_path, 'A_MCI_CN_allscantrain_split.csv', args.train_pct, is_train)
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'A_MCI_CN_allscantest_split.csv')
        else:
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'A_MCI_CN_allscanval_split.csv')

        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 2

    elif args.data_set == 'AD_MCI_CN':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = get_anno_path(args.data_path, 'AD_MCI_CN_allscantrain_split.csv', args.train_pct, is_train)
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'AD_MCI_CN_allscantest_split.csv')
        else:
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'AD_MCI_CN_allscanval_split.csv')

        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 3

    elif args.data_set == 'NIFD':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = get_anno_path(args.data_path, 'NIFD_allscans_train.csv', args.train_pct, is_train)
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'NIFD_allscans_test.csv')
        else:
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'NIFD_allscans_val.csv')

        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 2

    elif args.data_set == 'Kinetics-400':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv') 
        else:  
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'val.csv') 

        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 400
    
    elif args.data_set == 'SSV2':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv') 
        else:  
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'val.csv') 

        dataset = SSVideoClsDataset(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=1,
            num_segment=args.num_frames,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 174

    elif args.data_set == 'UCF101':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv') 
        else:  
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'val.csv') 

        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 101
    
    elif args.data_set == 'HMDB51':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv') 
        else:  
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'val.csv') 

        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 51

    elif args.data_set == 'IXI_age':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = get_anno_path(args.data_path, 'train_age_IXI_allscan.csv', args.train_pct, is_train)
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test_age_IXI_allscan.csv')
        else:
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'val_age_IXI_allscan.csv')

        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        # Age is a regression task, but we'll use nb_classes=1 to indicate regression
        # The actual number of classes doesn't matter for regression
        nb_classes = 1

    elif args.data_set == 'IXI_gender':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = get_anno_path(args.data_path, 'train_gender_IXI_allscan.csv', args.train_pct, is_train)
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test_gender_IXI_allscan.csv')
        else:
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'val_gender_IXI_allscan.csv')

        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 2

    else:
        raise NotImplementedError()
    assert nb_classes == args.nb_classes
    print("Number of the class = %d" % args.nb_classes)

    return dataset, nb_classes
