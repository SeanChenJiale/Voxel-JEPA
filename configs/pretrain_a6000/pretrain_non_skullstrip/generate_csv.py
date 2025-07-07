import csv
import glob

nii_list = glob.glob("/media/backup_16TB/sean/Monai/Pretraining/**/*.nii*",recursive=True)
print(len(nii_list))

final_list = []
for nii_path in nii_list:
    final_list.append([nii_path,0,-1])
    final_list.append([nii_path,1,-1])
    final_list.append([nii_path,2,-1])

# Write to a CSV file with whitespace as the delimiter
with open('/media/backup_16TB/sean/VJEPA/jepa/configs/pretrain_a6000/pretrain_non_skullstrip/non_skull_strip_nii_list.csv', 'w', newline='') as file:
    writer = csv.writer(file, delimiter=' ')
    writer.writerows(final_list)

print("Data written to 'non_skull_strip_nii_list.csv' with whitespace as the delimiter.")