#%%
import concurrent.futures
import numpy as np
import os
from preprocessing_utils import load_MNI_template, retrieve_nii_gz_paths ,bias_field_correction ,\
    rescale_intensity, equalize_hist , \
    load_nii , save_nii ,change_cwd , linear_interpolator, load_itk,\
    rigid_registration, filename_extract_from_path_with_ext, setup_logger, main_preprocessing,\
    json_find_folders_with_3d_acquisition, normalize_mri_to_rgb_float32, dataset_extractor
import SimpleITK as sitk
import pandas as pd
import glob
import pdb
import sys
change_cwd()
dataset_to_preprocess = "AIBL_nii"
output_folder_path = "/media/backup_16TB/sean/Monai/temp/AIBL_nii"
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path,exist_ok = True)
search_string="*MPRAGE_ADNI*.nii.gz"
# dataset_root_folder = "../Dataset/SOOP"
pathlist = glob.glob("/media/backup_16TB/sean/Monai/Dataset/AIBL_nii/**/*.nii.gz", recursive=True)
# pdb.set_trace()

# pathlist_2 = glob.glob("/media/backup_16TB/sean/Monai/PreProcessedData/WU1200/**/*.nii.gz", recursive=True)
# pathlist_3 = glob.glob("/media/backup_16TB/sean/Monai/PreProcessedData/ICBM/**/*.nii.gz", recursive=True)
# pathlist = pathlist + pathlist_2 #+ pathlist_3
# pathlist = json_find_folders_with_3d_acquisition(dataset_root_folder)
MNI_template, pathlist, finished_pathlist, finished_pp, undone_paths, logger = main_preprocessing(dataset_to_preprocess,search_string,pathlist = pathlist)

# pdb.set_trace()

# undone_paths = undone_paths[:8] ## for testing
# ## interject and call specific paths.
# undone_paths = pd.read_csv("/media/backup_16TB/sean/Monai/Dataset/NACC_filelist_filtered_t1_big_10mb.csv")["NIfTI File Path"].tolist()
# print(len(undone_paths))
# %%
def normalize_intensity_only(filepath):
    filename_to_save = filename_extract_from_path_with_ext(filepath)
    dataset = dataset_extractor(filepath)
    print(dataset)
    print(filename_to_save)
    temp_file = f"/media/backup_16TB/sean/Monai/NormalizedMRI225/{dataset}/{filename_to_save}"
    # Intensity normalization
    #check if directory for dataset exists, if not create it
    print(os.path.dirname(temp_file))
    if not os.path.exists(os.path.dirname(temp_file)):
        os.makedirs(os.path.dirname(temp_file))
    img, affine = load_nii(filepath)
    volume = normalize_mri_to_rgb_float32(img)
    save_nii(data=volume, affine=affine, path=temp_file)  # Save processed volume
    print(f"Normalized {filepath} and saved to {temp_file}")

def process_normalization(filepath):
    filename_to_save = filename_extract_from_path_with_ext(filepath)
    temp_file = f"../temp/{filename_to_save}_temp.nii.gz"
    # Intensity normalization
    img, affine = load_nii(filepath)
    volume = rescale_intensity(img, percentils=[0.5, 99.5], bins_num=256)
    volume = equalize_hist(volume, bins_num=256)
    save_nii(data=volume, affine=affine, path=temp_file)  # Save processed volume

#%%
def process_file(filepath):
    try:
        filename_to_save = filename_extract_from_path_with_ext(filepath)
        temp_file = f"../temp/{filename_to_save}_temp.nii.gz"
        img, affine = load_nii(filepath)

        img = bias_field_correction(img)
        save_nii(data=img, affine=affine, path=temp_file)

        # Linear interpolation
        img = load_itk(temp_file)
        img = linear_interpolator(img)
        sitk.WriteImage(img, temp_file)

        # Intensity normalization
        img, affine = load_nii(temp_file)
        volume = rescale_intensity(img, percentils=[0.5, 99.5], bins_num=256)
        volume = equalize_hist(volume, bins_num=256)
        save_nii(data=volume, affine=affine, path=temp_file)  # Save processed volume

        # Rigid registration
        img = load_itk(temp_file)
        img = rigid_registration(MNI_template, img)
        output_path = f"{output_folder_path}/{filename_to_save}"
        sitk.WriteImage(img, output_path)

        logger.info(f"{output_path} was saved.")
        # Remove the temporary file
        if os.path.exists(temp_file):
            os.remove(temp_file)
            logger.info(f"Temporary file {temp_file} was removed.")        
    except Exception as e:
        logger.error(f"Error processing {filepath}: {e}")

# Run processing in parallel
# when coding this, ensure that each thread's variables are independant of another thread.
def process_all_files_parallel(filepaths, num_workers=2):
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Map the processing function to the filepaths
            executor.map(process_file, filepaths)
    except KeyboardInterrupt:
        print("Terminating processes...")
        executor.shutdown(wait=False)  # Force shutdown without waiting for tasks to complete
        sys.exit("Script interrupted by user.")

# Example Usage
if __name__ == "__main__":
    # num_workers = os.cpu_count()  # Use all available CPU cores
    try:
        print('Trying to process files in parallel...')
        process_all_files_parallel(undone_paths, num_workers=2)
    except KeyboardInterrupt:
        print("Script interrupted by user.")
        sys.exit(1)
    # hd-bet -i IXI2 -o ../PreProcessedData/IXI2 --save_bet_mask
    #hd-bet -i ADNI_MP_RAGE -o ../PreProcessedData_tosend --save_bet_mask
    #hd-bet -i /media/backup_16TB/sean/Monai/temp/AIBL_nii -o /media/backup_16TB/sean/Monai/PreProcessedData/AIBL_nii -device cuda:1 --save_bet_mask
# %%
