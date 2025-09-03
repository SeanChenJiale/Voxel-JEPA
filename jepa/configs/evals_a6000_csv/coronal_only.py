import pandas as pd

df = pd.read_csv("/media/backup_16TB/sean/VJEPA/jepa/configs/evals_a6000_csv/eval_AD/AD_binary_train_alldata_nii_non_zoom.csv", header=None, sep=" ")

#keep if second column is 0
df = df[df[1] == 0]

df.to_csv("/media/backup_16TB/sean/VJEPA/jepa/configs/evals_a6000_csv/AD_binary_train_alldata_nii_non_zoom_coronal_only.csv", header=False, index=False, sep=" ")