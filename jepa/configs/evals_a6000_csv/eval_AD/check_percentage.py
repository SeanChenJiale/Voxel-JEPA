import pandas as pd

df_test = pd.read_csv("/media/backup_16TB/sean/VJEPA/jepa/configs/evals_a6000_csv/eval_AD/AD_binary_test_alldata_nii_non_zoom_coronal_only.csv", sep = ' ', header=None)
df_train = pd.read_csv("/media/backup_16TB/sean/VJEPA/jepa/configs/evals_a6000_csv/eval_AD/AD_binary_train_alldata_nii_non_zoom_coronal_only.csv", sep = ' ', header=None)

label_test = df_test[2]
print("Test set CN percentage: ", sum(label_test)/len(label_test))
print("Test set AD percentage: ", 1 - sum(label_test)/len(label_test))
label_train = df_train[2]
print("Train set CN percentage: ", sum(label_train)/len(label_train))
print("Train set AD percentage: ", 1 - sum(label_train)/len(label_train))
import pdb; pdb.set_trace()  # Put this where you want to pause execution
