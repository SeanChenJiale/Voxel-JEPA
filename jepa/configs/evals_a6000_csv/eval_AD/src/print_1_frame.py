import pandas as pd
import os
import tqdm
df = pd.read_csv("/media/backup_16TB/sean/Monai/src/FineTuningCSV_Generation/temp_nii_classes/AD_binary_test_alldata_mp4.csv", sep=' ', header=None)
#take first column as a list
df_path_list = df[0].tolist()
print(df_path_list[:1])
df_path_list = [path for path in df_path_list if os.path.exists(path)]
#from each mp4, extract the first frame and save it as a jpg
def extract_first_frame(video_path, output_path):
    import cv2
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Check if the video opened successfully
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return
    
    # Read the first frame
    ret, frame = cap.read()
    
    if ret:
        # Save the first frame as a JPEG file
        cv2.imwrite(output_path, frame)
        print(f"Saved first frame to {output_path}")
    else:
        print(f"Failed to read the first frame from {video_path}")
    
    # Release the video capture object
    cap.release()

for video_path in tqdm.tqdm(df_path_list):
    parent_folder_to_save = "/media/backup_16TB/sean/VJEPA/jepa/configs/evals_a6000_csv/eval_AD/src/debug_ad_bin_first_frame"
    #from video_path, extract the parent folder and append the video name
    if not os.path.exists(parent_folder_to_save):
        os.makedirs(parent_folder_to_save)
    output_path = os.path.join(parent_folder_to_save, f"{os.path.dirname(video_path).split('/')[-1]}_{os.path.basename(video_path)}.jpg")
    extract_first_frame(video_path, output_path)