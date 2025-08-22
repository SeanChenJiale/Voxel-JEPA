#%%
import csv
import glob
import csv
videolist = glob.glob("/media/backup_16TB/sean/Monai/PreProcessedDataDebug/pretraining/**/*.mp4",recursive=True)


#%%

# Output CSV file path
output_csv = "/media/backup_16TB/sean/VJEPA/jepa/configs/pretrain_a6000_csv/zoom48_mp4.csv"
# # Output CSV file paths
# train_csv = "/local_data/sean_hasitha/sean/VJEPA/jepa/Brainslice_96_Axisclassification_train.csv"
# val_csv = "/local_data/sean_hasitha/sean/VJEPA/jepa/Brainslice_96_Axisclassification_val.csv"
# Write to CSV
with open(output_csv, mode="w", newline="") as file:
    writer = csv.writer(file, delimiter=" ")  # Use space as the delimiter
    for video in videolist:
        if "Axial" in video:
            writer.writerow([video, 0])  # Write video path and integer 1
        elif "Coronal" in video:
            writer.writerow([video, 1])
        elif "Sagittal" in video:
            writer.writerow([video, 2])
# Print the path of the created CSV file
print(f"CSV file created at: {output_csv}")
# %%




# # Open the input CSV and split into train and val
# with open(input_csv, mode="r") as infile, \
#      open(train_csv, mode="w", newline="") as trainfile, \
#      open(val_csv, mode="w", newline="") as valfile:
    
#     reader = csv.reader(infile, delimiter=" ")  # Input uses space as delimiter
#     train_writer = csv.writer(trainfile, delimiter=" ")  # Train output uses space as delimiter
#     val_writer = csv.writer(valfile, delimiter=" ")  # Val output uses space as delimiter

#     for row in reader:
#         video_path = row[0]  # First column is the video path
#         if "/train/" in video_path:
#             train_writer.writerow(row)
#         elif "/val/" in video_path:
#             val_writer.writerow(row)

# # Print the paths of the created CSV files
# print(f"Train CSV file created at: {train_csv}")
# print(f"Validation CSV file created at: {val_csv}")
# # %%
