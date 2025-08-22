#%%
## run slice_extract first
import cv2
import os
import glob
import preprocessing_utils as preprocess
from natsort import natsorted  # Natural sorting
import logging

# Configure the logger
logging.basicConfig(
    level=logging.INFO,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s - %(levelname)s - %(message)s",  # Log format
    handlers=[
        logging.FileHandler("../logs/img_to_mp4_slices96.log"),  # Log to a file
        logging.StreamHandler()  # Log to the console
    ]
)
# Create a logger instance
logger = logging.getLogger(__name__)
preprocess.change_cwd()
nii_list = preprocess.retrieve_nii_gz_paths("../PreProcessData_/**/*.nii.gz")
parent_data_folder = "PreProcessData_"
parent_dir = "BrainSlice96_NACC"
nii_list = nii_list[:] ## take first 100 samples
print(len(nii_list))
print(nii_list[0])
#%%
count = 1
for nii_path in nii_list:
    print(f"currently at iteration : {count} of {len(nii_list)}")
    count += 1
    main_dataset = preprocess.dataset_extractor(nii_path,
                                                dataset_to_extract_after_string=parent_data_folder)
    listdir = preprocess.makedirs(nii_path,
                                  parent_directory_of_images = parent_dir,
                                  dataset_name = main_dataset,
                                  makedir=False)
    # Define the input and output paths
    Ordered_list = ["","Axial", "Coronal", "Sagittal"]
    print(Ordered_list[0])
    for listdir_curr_index in range(len(listdir)):
        ## it runs in alphabetical order, Axial, Coronal, Sagittal
        if listdir_curr_index == 0:
            continue
        curr_path = listdir[listdir_curr_index]
        print(curr_path)
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
        print(f"Video saved as {output_video}")
        logging.info(f"Video saved as {output_video}")

    # %%
