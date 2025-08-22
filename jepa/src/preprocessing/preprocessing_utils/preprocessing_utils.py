import SimpleITK as sitk
import glob
import os 
import logging
from scipy.signal import medfilt
import numpy as np
import nibabel as nib
import skimage
from skimage.transform import resize
import pandas as pd
from scipy.ndimage import affine_transform
import logging
import sys
import subprocess
import nibabel as nib
import os
import sys
import h5py
import numpy as np
import json

def main_preprocessing(main_datasetname, search_string = "*.nii.gz",pathlist = None):
    """ returns the MNI_template, temp_file, pathlist, finished_pathlist, finished_pp, undone_paths, logger""" 

    MNI_template = load_MNI_template()
    if pathlist is None:
        print("pathlist is not defined, finding search string's paths")
        pathlist = retrieve_nii_gz_paths(f'../Dataset/{main_datasetname}/**/{search_string}')

    finished_pathlist = retrieve_nii_gz_paths(f'../temp/{main_datasetname}/*.nii.gz')

    finished_pp= []

    for path in finished_pathlist:
        finished_pp.append(filename_extract_from_path_with_ext(path))
    
    finished_pp = finished_pathlist

    undone_paths = [path for path in pathlist if not any(sub in path for sub in finished_pp)]

    logger = setup_logger(f"../logs/{main_datasetname}preprocess.log")

    return MNI_template, pathlist, finished_pathlist, finished_pp, undone_paths, logger


def json_find_folders_with_3d_acquisition(base_dir):
    nii_files_with_3d = []

    # Walk through the directory structure
    for root, dirs, files in os.walk(base_dir):

        # Check each file in the directory
        for file in files:
            if file.endswith(".json"):  # Look for JSON files
                json_path = os.path.join(root, file)
                try:
                    # Open and parse the JSON file
                    with open(json_path, "r") as f:
                        data = json.load(f)
                    
                    # Check if "MRAcquisitionType" is "3D"
                    if data.get("MRAcquisitionType") == "3D":
                        # Replace .json with .nii.gz to find the corresponding file
                        nii_file = os.path.join(root, file.replace(".json", ".nii.gz"))
                        nii_files_with_3d.append(nii_file)
                except Exception as e:
                    print(f"Error reading {json_path}: {e}")
    
    print(f"Found {len(nii_files_with_3d)} .nii.gz files with 3D acquisition type.")
    return nii_files_with_3d

def convert_mnc_to_nifti(input_file, output_file=None):
    if not input_file.endswith('.mnc'):
        print("Error: Input file must have a '.mnc' extension.")
        sys.exit(1)
    if output_file is None:
        output_file = os.path.splitext(input_file)[0] + '.nii.gz'
    if not (output_file.endswith('.nii') or output_file.endswith('.nii.gz')):
        print("Error: Output file must have a '.nii' or '.nii.gz' extension.")
        sys.exit(1)
    try:
        img = nib.load(input_file)
        header = img.header
        data = img.get_fdata()
        if data.ndim == 4:
            with h5py.File(input_file, "r") as f:
                # MINC2 stores dimensions under 'minc-2.0/image/0/image'
                dimorder = f["minc-2.0/image/0/image"].attrs["dimorder"].decode()
                if dimorder.startswith("time,"):
                     data = np.transpose(data, (1, 2, 3, 0))
        affine = img.affine
        nifti_img = nib.Nifti1Image(data, affine)
        nifti_img.header.set_qform(affine)
        nifti_img.header.set_sform(affine)
        nifti_img.header['cal_min'] = data.min()
        nifti_img.header['cal_max'] = data.max()
        nib.save(nifti_img, output_file)
    except Exception as e:
        print(f"Error during conversion: {e}")
        sys.exit(1)

def setup_logger(logpath):
    # Ensure the directory exists
    log_dir = os.path.dirname(logpath)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Set to DEBUG to capture all levels

    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(logpath)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Avoid duplicate handlers
    if not logger.hasHandlers():
        logger.addHandler(file_handler)
        logger.addHandler(stdout_handler)

    return logger
def dataset_extractor(path,dataset_to_extract_after_string = "PreProcessedData") -> str:
    """
    returns the dir after PreProcessedData
    """
    listofpath = path.split("/")
    index = listofpath.index(dataset_to_extract_after_string)
    return listofpath[index + 1] 
def is_nii_gz_file(path):
    return path.endswith(".nii.gz")
def filename_extract_from_path_with_ext(path):
    """
    takes an nii.gz path and outputs only the filename.nii.gz
    """
    if is_nii_gz_file(path):
        return path.split("/")[-1]
    else:
        if path.endswith(".nii"):
            return path.split("/")[-1] + ".gz"
        else:
            print("Error: The provided path does not end with .nii or .nii.gz")
            return None


def file_remover_from_pathlist(pathlist,remove=False):
    """specify remove to be true to double check the files to be removed"""
    for file in pathlist:
        # Absolute filepath of the file to be removed
        file_to_remove = file

        try:
            # Check if the file exists
            if os.path.exists(file_to_remove):
                if remove:
                    os.remove(file_to_remove)
                    print(f"File {file_to_remove} has been removed.")
                else:
                    print(f"File {file_to_remove} will be removed.")
            else:
                print(f"File {file_to_remove} does not exist.")
        except Exception as e:
            print(f"An error occurred while trying to remove the file: {e}")

        print("All files have been removed.")
def find_dicom_folders(root_dir):
    """
    Recursively searches for folders containing DICOM files in the given directory.
    """

    dicom_folders = []
    
    for dirpath, _, filenames in os.walk(root_dir):
        if any(filename.lower().endswith(".dcm") for filename in filenames):
            dicom_folders.append(dirpath)  # Add folder if it contains .dcm files

    return dicom_folders

def makedirs(path,parent_directory_of_images,dataset_name,makedir = True) -> list:
    """
    creates a path based off splitting the relative path
    "../PreProcessedData/IXI/IXI002-Guys-0828-T1.nii.gz"
    and creates 3 separate folders, Axial, Coronal and Sagittal
    returns a list of 4 strings, [Main_path,Axial_path,Coronal_path,Sagittal_path]
    Could makedir = False to not create directories but still return the paths
    """
    folder_name = path.split("/")[-1].replace(".nii.gz","")
    
    Main_path = f"../{parent_directory_of_images}/{dataset_name}/{folder_name}"
    Axial_path = f"../{parent_directory_of_images}/{dataset_name}/{folder_name}/Axial"
    Coronal_path = f"../{parent_directory_of_images}/{dataset_name}/{folder_name}/Coronal"
    Sagittal_path = f"../{parent_directory_of_images}/{dataset_name}/{folder_name}/Sagittal"

    if makedir:
        os.makedirs(Main_path,exist_ok=True)
        os.makedirs(Axial_path,exist_ok=True)
        os.makedirs(Coronal_path,exist_ok=True)
        os.makedirs(Sagittal_path,exist_ok=True)

    return [Main_path,Axial_path,Coronal_path,Sagittal_path]

def change_cwd():
    """
    Gets the exact location of the current source file of this called function.
    """
    file = os.path.dirname(os.path.abspath(__file__))
    os.chdir(file)
    print("successfully changed cwd", os.getcwd())

def linear_interpolator(img):
     # Get original spacing and size
    original_spacing = img.GetSpacing() # input_image.GetSpacing()  # (dx, dy, dz)
    original_size = img.GetSize()        # (nx, ny, nz)

    # Define new isotropic spacing (1mm x 1mm x 1mm)
    new_spacing = (1.0, 1.0, 1.0)

    # Compute new size while preserving physical dimensions
    new_size = [
        int(round(original_size[i] * (original_spacing[i] / new_spacing[i])))
        for i in range(3)
    ]

    # Define resampling interpolator (use linear for intensity images, nearest for labels)
    interpolator = sitk.sitkLinear

    # Perform resampling
    resampled_image = sitk.Resample(
        img,
        new_size,
        sitk.Transform(),
        interpolator,
        img.GetOrigin(),
        new_spacing,
        img.GetDirection(),
        0,  # Default pixel value for areas outside original image
        img.GetPixelID()
    )

    return resampled_image


def retrieve_nii_gz_paths(search_path=None, print_len=True, recursive_bool=True, return_bet_nii = False) -> list:
    """
    Searches for .nii.gz files in the given directory.

    Parameters:
    - search_path (str): The search pattern (e.g., '../Dataset/OASIS3/OASIS3_1/**/*.nii.gz').
    - print_len (bool): Whether to print the number of files found.
    - recursive_bool (bool): Whether to search recursively.

    Returns:
    - list: Paths of found .nii.gz files.
    """

    if not search_path:
        print("Error: search_path is empty. Provide a valid path pattern.")
        return []

    try:
        # Get all .nii.gz files matching the search pattern
        search_list = glob.glob(search_path, recursive=recursive_bool)
        
        if return_bet_nii:
            print(f"{len(search_list)} with path {search_path} files found (including 'bet.nii.gz').")
            return search_list
        else:
            # Exclude files with 'bet.nii.gz' in their names
            filtered_list = [path for path in search_list if 'bet.nii.gz' not in path]

            if print_len:
                print(f"{len(filtered_list)} with path {search_path} files found (excluding 'bet.nii.gz').")
            return filtered_list
    except Exception as e:
        print(f"Error in loading: {e}")
        return []

def denoise(volume, kernel_size=3):
    return medfilt(volume, kernel_size)

def load_MNI_template(path_to_MNI = "/media/backup_16TB/sean/Monai/Dataset/MNI/MNI152_T1_1mm.nii.gz") -> sitk.ReadImage:
    """
    takes in a path_to_MNI, '../Dataset/MNI/MNI152_T1_1mm.nii.gz'
    """
    try :
        fixed_image = sitk.ReadImage(path_to_MNI)  # MNI template
        print("load success")
    except: 
        print("could not load the template. Check the file pathing")
    return fixed_image
def check_image_orientation(reoriented_file_path,verbose_print = True):
        
    # Load the reoriented NIfTI file
    reoriented_img = nib.load(reoriented_file_path)

    # Get the current affine matrix
    affine = reoriented_img.affine
    if verbose_print :
        print("Current Affine Matrix:")
        print(affine)

    # Check the orientation
    orientation = nib.aff2axcodes(reoriented_img.affine)
    if verbose_print :
        print("Orientation:", orientation)
    return orientation
def rigid_registration(template,moving_image):
    
    # Convert images to float32
    fixed_image_cast = sitk.Cast(template, sitk.sitkFloat32)
    moving_image = sitk.Cast(moving_image, sitk.sitkFloat32)
    initial_transform=sitk.CenteredTransformInitializer(fixed_image_cast,moving_image,sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY)
    moving_resampled = sitk.Resample(moving_image, fixed_image_cast, initial_transform, sitk.sitkLinear, 0.0, moving_image.GetPixelID())
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsCorrelation()
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)
    registration_method.SetInterpolator(sitk.sitkNearestNeighbor) ##update on this. It performs better in terms of registration.
    registration_method.SetOptimizerAsGradientDescent(learningRate=0.1, numberOfIterations=500)
    registration_method.SetOptimizerScalesFromPhysicalShift()
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2,1,0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    registration_method.SetInitialTransform(initial_transform, inPlace=False)
    final_transform = registration_method.Execute(sitk.Cast(fixed_image_cast, sitk.sitkFloat32), sitk.Cast(moving_image, sitk.sitkFloat32))
    moving_resampled = sitk.Resample(moving_image, fixed_image_cast, final_transform, sitk.sitkLinear, 0.0, moving_image.GetPixelID())
    return moving_resampled

def bias_field_correction(img):
    image = sitk.GetImageFromArray(img)
    maskImage = sitk.OtsuThreshold(image, 0, 1, 200)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    numberFittingLevels = 4

    corrector.SetMaximumNumberOfIterations([100] * numberFittingLevels)
    corrected_image = corrector.Execute(image, maskImage)
    log_bias_field = corrector.GetLogBiasFieldAsImage(image)
    # Convert to a float type for better correction
    log_bias_field = sitk.Cast(log_bias_field, sitk.sitkFloat64)
    corrected_image_full_resolution = image / sitk.Exp(log_bias_field)
    return sitk.GetArrayFromImage(corrected_image_full_resolution)

def rescale_intensity(volume, percentils=[0.5, 99.5], bins_num=256):
    #remove background pixels by the otsu filtering
    t = skimage.filters.threshold_otsu(volume,nbins=6)
    volume[volume < t] = 0
    
    obj_volume = volume[np.where(volume > 0)]
    min_value = np.percentile(obj_volume, percentils[0])
    max_value = np.percentile(obj_volume, percentils[1])
    if bins_num == 0:
        obj_volume = (obj_volume - min_value) / (max_value - min_value).astype(np.float32)
    else:
        obj_volume = np.round((obj_volume - min_value) / (max_value - min_value) * (bins_num - 1))
        obj_volume[np.where(obj_volume < 1)] = 1
        obj_volume[np.where(obj_volume > (bins_num - 1))] = bins_num - 1

    volume = volume.astype(obj_volume.dtype)
    volume[np.where(volume > 0)] = obj_volume
    return volume

def normalize_mri_to_rgb_float32(volume):
    """
    Normalize MRI volume to the range 0-255 (similar to RGB video frames) while keeping float32 data type.

    Parameters:
    - volume (numpy.ndarray): The input MRI volume.

    Returns:
    - numpy.ndarray: The normalized MRI volume as float32.
    """
    # Ensure the volume is a numpy array and convert to float32
    volume = np.asarray(volume, dtype=np.float32)

    # Find the minimum and maximum intensity values
    min_val = np.min(volume)
    max_val = np.max(volume)

    # Normalize the volume to the range 0-255
    normalized_volume = (volume - min_val) / (max_val - min_val) * 255.0

    return normalized_volume
    
# equalize the histogram of the image
def equalize_hist(volume, bins_num=256):
    obj_volume = volume[np.where(volume > 0)]
    hist, bins = np.histogram(obj_volume, bins_num)
    cdf = hist.cumsum()
    cdf = (bins_num - 1) * cdf / cdf[-1]

    obj_volume = np.round(np.interp(obj_volume, bins[:-1], cdf)).astype(obj_volume.dtype)
    volume[np.where(volume > 0)] = obj_volume
    return volume

# enhance the image
def enhance(volume, kernel_size=3,
            percentils=[0.5, 99.5], bins_num=256, eh=True):
    try:
        volume = bias_field_correction(volume)
        volume = denoise(volume, kernel_size)
        volume = rescale_intensity(volume, percentils, bins_num)
        if eh:
            volume = equalize_hist(volume, bins_num)
        return volume
    except RuntimeError:
        logging.warning('Failed enchancing')

def load_nii(path):
    nii = nib.load(path)
    return nii.get_fdata(), nii.affine

def load_itk(path):
    input_image = sitk.ReadImage(path)
    return input_image


def save_nii(data, path, affine):
    nib.save(nib.Nifti1Image(data, affine), path)
    return