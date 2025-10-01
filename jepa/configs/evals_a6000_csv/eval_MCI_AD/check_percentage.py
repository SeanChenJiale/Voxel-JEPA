import pandas as pd

df_test = pd.read_csv("/media/backup_16TB/sean/VJEPA/jepa/configs/evals_a6000_csv/eval_MCI_AD/AD_MCI_test.csv", sep = ' ', header=None)
df_train = pd.read_csv("/media/backup_16TB/sean/VJEPA/jepa/configs/evals_a6000_csv/eval_MCI_AD/AD_MCI_train.csv", sep = ' ', header=None)

label_test = df_test[2]
print("count AD: ", sum(label_test))
print("count MCI: ", len(label_test) - sum(label_test))
print("Test set MCI percentage: ", sum(label_test)/len(label_test))
print("Test set AD percentage: ", 1 - sum(label_test)/len(label_test))
print("Test size: ", len(label_test))
label_train = df_train[2]
print("count AD: ", sum(label_train))
print("count MCI: ", len(label_train) - sum(label_train))
print("Train set MCI percentage: ", sum(label_train)/len(label_train))
print("Train set AD percentage: ", 1 - sum(label_train)/len(label_train))
print("Train size: ", len(label_train))
# import pdb; pdb.set_trace()  # Put this where you want to pause execution
print("Total = ", len(label_test) + len(label_train))
print("Total AD: ",sum(label_test) + sum(label_train))
print("Total MCI: ",len(label_test) - sum(label_test) + len(label_train) - sum(label_train))
