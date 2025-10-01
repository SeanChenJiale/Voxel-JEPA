import pandas as pd

df_test = pd.read_csv("/media/backup_16TB/sean/VJEPA/jepa/configs/evals_a6000_csv/eval_AD_MCI_CN/AD_CN_MCI_test.csv", sep = ' ', header=None)
df_train = pd.read_csv("/media/backup_16TB/sean/VJEPA/jepa/configs/evals_a6000_csv/eval_AD_MCI_CN/AD_CN_MCI_train.csv", sep = ' ', header=None)

#data set has 3 labels. 0 = CN, 1 = MCI, 2 = AD
label_test = df_test[2]
print("Test set CN percentage: ", sum(label_test==0)/len(label_test))
print("Test set MCI percentage: ", sum(label_test==1)/len(label_test))
print("Test set AD percentage: ", sum(label_test==2)/len(label_test))

label_train = df_train[2]
print("Train set CN percentage: ", sum(label_train==0)/len(label_train))
print("Train set MCI percentage: ", sum(label_train==1)/len(label_train))
print("Train set AD percentage: ", sum(label_train==2)/len(label_train))
import pdb; pdb.set_trace()  # Put this where you want to pause execution
