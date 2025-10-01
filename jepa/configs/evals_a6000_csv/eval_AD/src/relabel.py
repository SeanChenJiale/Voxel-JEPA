import pandas as pd

k400_test_df = pd.read_csv("/media/backup_16TB/sean/VJEPA/jepa/configs/evals_a6000_csv/k400_test.csv", sep=' ', header=None)
k400_train_df = pd.read_csv("/media/backup_16TB/sean/VJEPA/jepa/configs/evals_a6000_csv/k400_train.csv", sep=' ', header=None)
AD_test_df = pd.read_csv("/media/backup_16TB/sean/VJEPA/jepa/configs/evals_a6000_csv/eval_AD/AD_binary_test_alldata_mp4.csv", sep=' ', header=None)
AD_train_df = pd.read_csv("/media/backup_16TB/sean/VJEPA/jepa/configs/evals_a6000_csv/eval_AD/AD_binary_train_alldata_mp4.csv", sep=' ', header=None)

#swap the whole column of AD to 0
AD_test_df[1] = 0
AD_train_df[1] = 0


combined_test_df = pd.concat([k400_test_df, AD_test_df], ignore_index=True)
combined_train_df = pd.concat([k400_train_df, AD_train_df], ignore_index=True)

combined_test_df.to_csv("/media/backup_16TB/sean/VJEPA/jepa/configs/evals_a6000_csv/eval_AD/k400_AD_test.csv", sep=' ', index=False, header=False)
combined_train_df.to_csv("/media/backup_16TB/sean/VJEPA/jepa/configs/evals_a6000_csv/eval_AD/k400_AD_train.csv", sep=' ', index=False, header=False)