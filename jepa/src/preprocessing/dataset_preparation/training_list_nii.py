#%%
import csv
import glob

nii_list = glob.glob("/media/backup_16TB/sean/Monai/PreProcessedData_box/pretraining/**/*.nii.gz",recursive=True)


#%%

# Output CSV file path
output_csv = "/media/backup_16TB/sean/VJEPA/jepa/configs/pretrain_a6000_csv/pretraining_nii_non_zoom.csv"
with open(output_csv, mode="w", newline="") as file:
    writer = csv.writer(file, delimiter=" ")   
    for nii in nii_list:
         # Use space as the delimiter path, axis, label
        writer.writerow([nii, 0, 0])
        writer.writerow([nii, 1, 0])
        writer.writerow([nii, 2, 0])

# Print the path of the created CSV file
print(f"CSV file created at: {output_csv}")

# %%
