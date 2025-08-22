import SimpleITK as sitk
import glob
import tqdm
from pathlib import Path
import SimpleITK as sitk
import pdb
import tqdm
# nii_list = glob.glob("/media/backup_16TB/sean/Monai/PreProcessedData/SALD/**/*T1*.nii.gz",recursive = True) 
# nii_list = nii_list + glob.glob("/media/backup_16TB/sean/Monai/PreProcessedData/NACC_all_T1_3D/**/*.nii.gz",recursive = True)
# nii_list = nii_list + glob.glob("/media/backup_16TB/sean/Monai/PreProcessedData/NACC_MPRAGE/**/*.nii.gz",recursive = True)
# nii_list = nii_list + glob.glob("/media/backup_16TB/sean/Monai/PreProcessedData/OASIS/**/*.nii.gz",recursive = True)
# nii_list = nii_list + glob.glob("/media/backup_16TB/sean/Monai/PreProcessedData/ICBM/**/*.nii.gz",recursive = True)
# nii_list = nii_list + glob.glob("/media/backup_16TB/sean/Monai/PreProcessedData/NACC_T1_Volume_mri/**/*.nii.gz",recursive = True)
# nii_list = nii_list + glob.glob("/media/backup_16TB/sean/Monai/PreProcessedData/AIBL_nii/**/*.nii.gz",recursive = True)
# nii_list = nii_list + glob.glob("/media/backup_16TB/sean/Monai/PreProcessedData/ADNI_MP_RAGE/**/*.nii.gz",recursive = True)
nii_list = glob.glob("/media/backup_16TB/sean/Monai/PreProcessedData/SOOP/**/*T1*.nii.gz",recursive = True)
nii_list = nii_list + glob.glob("/media/backup_16TB/sean/Monai/PreProcessedData/GSP/**/*.nii.gz",recursive = True)
nii_list = nii_list + glob.glob("/media/backup_16TB/sean/Monai/PreProcessedData/AOMIC/**/*T1w*.nii.gz",recursive = True)
nii_list = nii_list + glob.glob("/media/backup_16TB/sean/Monai/PreProcessedData/WU1200/**/*T1w*.nii.gz",recursive = True)

# nii_list = glob.glob("/media/backup_16TB/sean/Monai/PreProcessedData/ADNI_MPRAGE/**/*.nii.gz",recursive = True)
error_list = []
# nii_list = nii_list[:2] #to debug
for nii_path in tqdm.tqdm(nii_list):

    output_path = nii_path.replace("/media/backup_16TB/sean/Monai/PreProcessedData/","/media/backup_16TB/sean/Monai/PreProcessedDataZoomed/pretraining/")
    output_path = Path(output_path)  # or .nii, .nii.gz, etc.
    if not output_path.exists():
        # Ensure directory exists (mkdir -p equivalent)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        #Load your skull-stripped 3D image
        img = sitk.ReadImage(nii_path)

        #Create a binary mask of the brain
        mask = sitk.BinaryThreshold(img, lowerThreshold=1e-6, upperThreshold=1e9, insideValue=1, outsideValue=0)

        #Find the axis-aligned bounding box via LabelShapeStatisticsImageFilter
        stats = sitk.LabelShapeStatisticsImageFilter()
        stats.Execute(mask)
        if not stats.HasLabel(1):
            error_list.append(output_path)
            print(output_path, "  Mask contains no foreground voxels")
            continue
            raise RuntimeError("Mask contains no foreground voxels")

        x0, y0, z0, sx, sy, sz = stats.GetBoundingBox(1)
        box_size = (sx, sy, sz)

        crop = sitk.RegionOfInterest(img, size=box_size, index=(x0, y0, z0))

        desired = (224, 224, 224)
        orig_size = crop.GetSize()      # (sx, sy, sz)
        orig_spacing = crop.GetSpacing()

        scale = min(d / s for d, s in zip(desired, orig_size))
        new_size = [int(round(s * scale)) for s in orig_size]
        new_spacing = [osp / scale for osp in orig_spacing]

        resampled = sitk.Resample(
            crop,
            new_size,
            sitk.Transform(),         # identity (no rotation/translation)
            sitk.sitkLinear,          # best for intensity images
            crop.GetOrigin(),
            new_spacing,
            crop.GetDirection(),
            0,                        # default value outside if needed
            crop.GetPixelID()
        )

        # Compute symmetric padding to center the brain
        pad_lower = [ (d - n)//2 for d, n in zip(desired, new_size) ]
        pad_upper = [ d - n - low for d, n, low in zip(desired, new_size, pad_lower) ]

        final = sitk.ConstantPad(resampled, pad_lower, pad_upper, constant=0)


        #Write out padded image; spacing, origin, direction are correctly handled
        sitk.WriteImage(final, str(output_path))

    else:
        print(f"Output file already exists, skipping creation: {output_path}") 
        