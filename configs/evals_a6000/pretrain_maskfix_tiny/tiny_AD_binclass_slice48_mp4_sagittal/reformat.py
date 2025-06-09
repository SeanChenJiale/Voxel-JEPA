import pandas as pd

df = pd.read_csv("/media/backup_16TB/sean/VJEPA/jepa/configs/evals_a6000/tiny_AD_binclass_slice48_mp4/Brainslice_48_ADclassification_train_final.csv",
                     sep=' ', header=None, names=['path', 'dummy'])
df_val = pd.read_csv("/media/backup_16TB/sean/VJEPA/jepa/configs/evals_a6000/tiny_AD_binclass_slice48_mp4/Brainslice_48_ADclassification_val_final.csv",
                     sep=' ', header=None, names=['path', 'dummy'])
# filter dummy == 0 and == 5

df = df[df['dummy'].isin([0, 5])]
# change 5 to 1 in dummy column
df['dummy'] = df['dummy'].replace(5, 1)
df.to_csv("/media/backup_16TB/sean/VJEPA/jepa/configs/evals_a6000/tiny_AD_binclass_slice48_mp4/Brainslice_48_ADclassification_train_final_filtered.csv",
          index=False, header=False, sep=' ')




df_val = df_val[df_val['dummy'].isin([0, 5])]
df_val['dummy'] = df_val['dummy'].replace(5, 1)
df_val.to_csv("/media/backup_16TB/sean/VJEPA/jepa/configs/evals_a6000/tiny_AD_binclass_slice48_mp4/Brainslice_48_ADclassification_val_final_filtered.csv",
              index=False, header=False, sep=' ')