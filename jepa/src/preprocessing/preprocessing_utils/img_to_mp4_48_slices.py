#%%
## run slice_extract first
import cv2
import os
import glob
import preprocessing_utils as preprocess
from natsort import natsorted  # Natural sorting
import logging
import tqdm
# Configure the logger
logging.basicConfig(
    level=logging.INFO,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s - %(levelname)s - %(message)s",  # Log format
    handlers=[
        logging.FileHandler("/media/backup_16TB/sean/Monai/logs/img_to_mp4_slices48_15_aug.log"),  # Log to a file
        logging.StreamHandler()  # Log to the console
    ]
)
# Create a logger instance
logger = logging.getLogger(__name__)
preprocess.change_cwd()
# nii_list = ["/media/backup_16TB/sean/Monai/PreProcessedDataZoomed/evals/ADNI_MPRAGE_nii/037_S_4879_20250410095731/2013-09-09_13_41_18.0/2013-09-09_13_41_18.0_MPRAGE_20130909134118_2.nii.gz"]
nii_list = preprocess.retrieve_nii_gz_paths("/media/backup_16TB/sean/Monai/PreProcessedDataZoomed/pretraining/**/*.nii*")
# nii_list_1 = preprocess.retrieve_nii_gz_paths("/media/backup_16TB/sean/Monai/PreProcessedData/MCSA/**/*.nii*")
# nii_list_2 = preprocess.retrieve_nii_gz_paths("/media/backup_16TB/sean/Monai/PreProcessedData/**/*unstripped.nii.gz")
# nii_list = nii_list + nii_list_1 + nii_list_2
original_data_folder = "/media/backup_16TB/sean/Monai/PreProcessedDataZoomed"
output_folder = "/media/backup_16TB/sean/Monai/PreProcessedDataDebug"
nii_list = nii_list[:] ## take first 100 samples
# print(len(nii_list))
# print(nii_list[0])
#%%
for nii_path in tqdm.tqdm(nii_list):
    temp = nii_path.replace(".nii", "")
    temp = temp.replace(".gz", "")
    output_niibasefolder_path = temp.replace(original_data_folder, output_folder)
    listdir = [output_niibasefolder_path,
                         f"{output_niibasefolder_path}/Axial",
                         f"{output_niibasefolder_path}/Coronal",
                         f"{output_niibasefolder_path}/Sagittal"]
    # Define the input and output paths
    Ordered_list = ["","Axial", "Coronal", "Sagittal"]

    for listdir_curr_index in range(len(listdir)):
        ## it runs in alphabetical order, Axial, Coronal, Sagittal
        if listdir_curr_index == 0:
            continue
        curr_path = listdir[listdir_curr_index]
        # print(curr_path)
        output_video = f"{listdir[0]}/{Ordered_list[listdir_curr_index]}.mp4"

        # Get all images (Assumes they are PNG/JPG and sorted in order)
        image_files = natsorted(glob.glob(os.path.join(curr_path, "*.png")))  # Change to "*.jpg" if needed

        # Ensure there are exactly 48 slices
        if len(image_files) != 48:
            print(f"Error: Expected 48 slices, but found {len(image_files)} in {curr_path}. Skipping...")
            continue

        # Read first image to get dimensions
        first_img = cv2.imread(image_files[0], cv2.IMREAD_GRAYSCALE)  # Read in grayscale
        height, width = first_img.shape

        # Define video writer (MP4, 30 FPS, same resolution as images)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for MP4
        video_writer = cv2.VideoWriter(output_video, fourcc, 30, (width, height), isColor=True)

        # Split slices into R, G, B groups
        red_slices = image_files[::3]  # alternate every 3 images
        green_slices = image_files[1::3]  # Next 16 slices
        blue_slices = image_files[2::3]  # Last 16 slices

        # Loop through slices and create RGB frames
        for i in range(16):  # Loop through the first 16 slices (one frame per iteration)
            # Read slices for each channel
            red_img = cv2.imread(red_slices[i], cv2.IMREAD_GRAYSCALE)  # Red channel
            green_img = cv2.imread(green_slices[i], cv2.IMREAD_GRAYSCALE)  # Green channel
            blue_img = cv2.imread(blue_slices[i], cv2.IMREAD_GRAYSCALE)  # Blue channel

            # Stack into an RGB image
            rgb_img = cv2.merge((blue_img, green_img, red_img))  # OpenCV uses BGR order

            # Repeat each frame 4 times
            for _ in range(4):  # Repeat each frame 4 times
                video_writer.write(rgb_img)

        # Release the video writer
        video_writer.release()
        logging.info(f"Video saved as {output_video}")

    # %%
