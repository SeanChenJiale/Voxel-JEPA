import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("/media/backup_16TB/sean/VJEPA/jepa/configs/evals_a6000_csv/eval_age/IXI_test_T1w.csv",header=None,sep=" ")
df.iloc[:,2] = df.iloc[:,2].astype(float)
mean_age = df.iloc[:,2].mean()
print(f"Mean age in test set: {mean_age}")
plt.hist(df.iloc[:,2], bins=20, edgecolor='black')
plt.title('Age Distribution in Test Set')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.savefig("/media/backup_16TB/sean/VJEPA/jepa/configs/evals_a6000_csv/eval_age/age_distribution_test_set.png")


df = pd.read_csv("/media/backup_16TB/sean/VJEPA/jepa/configs/evals_a6000_csv/eval_age/IXI_train_T1w.csv",header=None,sep=" ")
df.iloc[:,2] = df.iloc[:,2].astype(float)
mean_age = df.iloc[:,2].mean()
print(f"Mean age in train set: {mean_age}")


plt.hist(df.iloc[:,2], bins=20, edgecolor='black')
plt.title('Age Distribution in Training Set')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.savefig("/media/backup_16TB/sean/VJEPA/jepa/configs/evals_a6000_csv/eval_age/age_distribution_train_set.png")
plt.clf()

