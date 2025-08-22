import os
import subprocess
import preprocessing_utils
import pdb

preprocessing_utils.change_cwd()
# Define input and output directories
input_dir = "/media/backup_16TB/sean/Monai/Dataset/ABIL"  # Replace with the path to your DICOM folders
output_dir = "../Dataset/ABIL_nii"  # Replace with the path to save NIfTI files
os.makedirs(output_dir, exist_ok=True)

for topfolder in os.listdir(input_dir):
# Ensure the output directory exists
    # Loop through each folder in the input directory
    for folder in os.listdir(os.path.join(input_dir,topfolder)):
        dicom_folder_path = os.path.join(input_dir,topfolder, folder)
        output_folder_path = os.path.join(output_dir,topfolder, folder)
        pdb.set_trace()
        # Check if it's a directory
        break
        if os.path.isdir(dicom_folder_path):
            os.makedirs(output_folder_path, exist_ok=True)  # Create output folder

            # Run dcm2niix command
            command = [
                "/usr/bin/dcm2niix",
                "-z", "y",  # Compress output to .nii.gz
                "-o", output_folder_path,  # Output folder
                dicom_folder_path  # Input DICOM folder
            ]
            subprocess.run(command, check=True)
    break
    print("Conversion completed!")