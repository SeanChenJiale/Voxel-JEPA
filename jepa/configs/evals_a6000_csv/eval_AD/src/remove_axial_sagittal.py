import pandas as pd 

mp4_traindata = pd.read_csv('/media/backup_16TB/sean/VJEPA/jepa/configs/evals_a6000_csv/eval_AD/AD_binary_test_alldata_mp4.csv', sep =' ', header=None)
mp4_testdata = pd.read_csv('/media/backup_16TB/sean/VJEPA/jepa/configs/evals_a6000_csv/eval_AD/AD_binary_test_alldata_mp4.csv', sep =' ', header=None)

# For filepaths in column 1, remove any rows that contain 'axial' or 'sagittal'
mp4_traindata = mp4_traindata[~mp4_traindata[0].str.contains('Axial')]
mp4_traindata = mp4_traindata[~mp4_traindata[0].str.contains('Sagittal')]
mp4_traindata.to_csv('/media/backup_16TB/sean/VJEPA/jepa/configs/evals_a6000_csv/eval_AD/AD_binary_test_alldata_mp4_no_axial_sagittal.csv', sep=' ', header=False, index=False)
mp4_testdata = mp4_testdata[~mp4_testdata[0].str.contains('Axial')]
mp4_testdata = mp4_testdata[~mp4_testdata[0].str.contains('Sagittal')]
mp4_testdata.to_csv('/media/backup_16TB/sean/VJEPA/jepa/configs/evals_a6000_csv/eval_AD/AD_binary_test_alldata_mp4_no_axial_sagittal.csv', sep=' ', header=False, index=False)