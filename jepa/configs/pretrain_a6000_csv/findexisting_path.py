import pandas as pd
import os
df = pd.read_csv("/media/backup_16TB/sean/VJEPA/jepa/configs/evals_a6000_csv/eval_AD/AD_binary_test_alldata_nii_non_zoom.csv", sep = ' ', header=None)

#return first column 
def find_existing_path():
    return df.iloc[:, 0].tolist()

find_existing_path = find_existing_path()

filesnotfound = []
for path in find_existing_path:
    if not os.path.exists(path):
        print(f"Path does not exist: {path}")
        filesnotfound.append(path)
    else:
        pass
print(f"Total files not found: {len(filesnotfound)}")
print(f"Files not found: {filesnotfound}")

#remove lines in df where filesnotfound
df = df[~df.iloc[:, 0].isin(filesnotfound)]
# Save the updated DataFrame back to CSV
if len(filesnotfound) > 0:
    print(f"Saving updated CSV with {len(filesnotfound)} files removed.")
    df.to_csv("/media/backup_16TB/sean/VJEPA/jepa/configs/evals_a6000_csv/eval_AD/AD_binary_test_alldata_nii_non_zoom_update.csv", sep=' ', header=False, index=False)