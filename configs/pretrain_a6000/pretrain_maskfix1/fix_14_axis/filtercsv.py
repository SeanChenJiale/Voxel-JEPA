import pandas as pd

df = pd.read_csv("/media/backup_16TB/sean/VJEPA/jepa/configs/pretrain_a6000/pretrain_maskfix1/pretrain3_zoom.csv",sep = ' ',header = None)

# Filter rows where 'City' contains 'New' (case-insensitive)
filtered_df = df[df[0].str.contains(r'Axial', case=False, na=False)]
filtered_df.to_csv("/media/backup_16TB/sean/VJEPA/jepa/configs/pretrain_a6000/pretrain_maskfix1/fix_14_axis/pretrain3_axial.csv",index=False,header=None,sep=' ')

filtered_df = df[df[0].str.contains(r'Sagittal', case=False, na=False)]
filtered_df.to_csv("/media/backup_16TB/sean/VJEPA/jepa/configs/pretrain_a6000/pretrain_maskfix1/fix_14_axis/pretrain3_sagittal.csv",index=False,header=None,sep=' ')

filtered_df = df[df[0].str.contains(r'Coronal', case=False, na=False)]
filtered_df.to_csv("/media/backup_16TB/sean/VJEPA/jepa/configs/pretrain_a6000/pretrain_maskfix1/fix_14_axis/pretrain3_coronal.csv",index=False,header=None,sep=' ')