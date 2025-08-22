import pandas as pd

df = pd.read_csv("/media/backup_16TB/sean/VJEPA/jepa/configs/evals_a6000_csv/eval_AD/AD_binary_train_alldata_mp4.csv",
sep =' ', header = None)
print(df.head())

# Group by the second column and count the number of rows in each group, then divide by 3
group_counts = df.groupby(1).size() // 3
print(group_counts)