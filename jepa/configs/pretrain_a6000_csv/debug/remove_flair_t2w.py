import pandas as pd

csv_path = "/media/backup_16TB/sean/VJEPA/jepa/configs/pretrain_a6000_csv/pretraining_nii_box.csv"

# Read the CSV file
df = pd.read_csv(csv_path, header=None, delimiter=" ")

# Filter rows where the first column does NOT contain FLAIR, t2, or T1Gd
mask = ~df[0].str.contains("FLAIR") & ~df[0].str.contains("t2") & ~df[0].str.contains("T1Gd")
filtered_df = df[mask]

# Save the filtered DataFrame (all columns) to a new CSV file
output_csv = "/media/backup_16TB/sean/VJEPA/jepa/configs/pretrain_a6000_csv/pretraining_nii_box_filtered.csv"
filtered_df.to_csv(output_csv, index=False, header=False, sep=" ")

print(f"Filtered CSV file created at: {output_csv}")
print(f"Number of paths before filtering: {len(df)}")
print(f"Number of paths after filtering: {len(filtered_df)}")
