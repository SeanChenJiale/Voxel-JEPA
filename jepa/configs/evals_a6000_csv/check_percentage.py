import pandas as pd

df_test = pd.read_csv("/media/backup_16TB/sean/VJEPA/jepa/configs/evals_a6000_csv/eval_AD/AD_binary_test_alldata_nii_non_zoom_coronal_only.csv", sep = ' ', header=None)
df_train = pd.read_csv("/media/backup_16TB/sean/VJEPA/jepa/configs/evals_a6000_csv/eval_AD/AD_binary_train_alldata_nii_non_zoom_coronal_only.csv", sep = ' ', header=None)
label_0 = "CN"
label_1 = "AD"
label_test = df_test[2]
print(f"count {label_1}: ", sum(label_test))
print(f"count {label_0}: ", len(label_test) - sum(label_test))
print(f"Test set {label_0} percentage: ", sum(label_test)/len(label_test))
print(f"Test set {label_1} percentage: ", 1 - sum(label_test)/len(label_test))
print("Test size: ", len(label_test))
label_train = df_train[2]
print(f"count {label_1}: ", sum(label_train))
print(f"count {label_0}: ", len(label_train) - sum(label_train))
print(f"Train set {label_0} percentage: ", sum(label_train)/len(label_train))
print(f"Train set {label_1} percentage: ", 1 - sum(label_train)/len(label_train))
print("Train size: ", len(label_train))
# import pdb; pdb.set_trace()  # Put this where you want to pause execution
print("Total = ", len(label_test) + len(label_train))
print(f"Total {label_1}: ",sum(label_test) + sum(label_train))
print(f"Total {label_0}: ",len(label_test) - sum(label_test) + len(label_train) - sum(label_train))
