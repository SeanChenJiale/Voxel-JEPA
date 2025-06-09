import torch 
import pandas as pd
import re
import ast
import os
import matplotlib.pyplot as plt
import numpy as np
from mask_visualization_within_epoch import plot_mask_visualization
def reshape_tensor(hi,zi):
    assert hi.shape == zi.shape
    hi = hi.reshape(hi.size(-3),hi.size(-1),hi.size(-2)) #[clip,dim,pred_patch_value]
    zi = zi.reshape(hi.size(-3),zi.size(-1),zi.size(-2)) #[clip,dim,pred_patch_value]
    return hi,zi

def load_tensor(hi_path,zi_path):
    return reshape_tensor(torch.load(hi_path),torch.load(zi_path))

def load_csv(csvpath):
    return pd.read_csv(csvpath)

def extract_mask_number(file_name):
    # Use regex to find the number between "mask" and ".pt"
    match = re.search(r'mask(\d+)\.pt', file_name)
    if match:
        return int(match.group(1))  # Convert the extracted number to an integer
    return None  # Return None if no match is found

    
def format_hi_zi_to_list(hi_path,zi_path,patch_index_list,pred_dim_index = 0,clip_index = 0):
    """
    returns hi_patch_value_list,zi_patch_value_list,patch_index_list
    """
    def select_clip(hi_tensor,zi_tensor,clip_index):
        return hi_tensor[clip_index], zi_tensor[clip_index]
    hi_0, zi_0 = load_tensor(hi_path,zi_path)
    hi_output = [0 for _ in range(1568)]
    zi_output = [0 for _ in range(1568)]
    hi_zi_diff_output = [0 for _ in range(1568)]

    hi_clip,zi_clip = select_clip(hi_0,zi_0,clip_index)
    pred_patch_value_hi_list = hi_clip[pred_dim_index].tolist()
    pred_patch_value_zi_list = zi_clip[pred_dim_index].tolist()

    for value_hi, value_zi, patch_index in zip(pred_patch_value_hi_list,pred_patch_value_zi_list,patch_index_list):
        hi_output[patch_index] = value_hi
        zi_output[patch_index] = value_zi
        hi_zi_diff_output[patch_index] = abs(value_hi-value_zi)
    return hi_output,zi_output,hi_zi_diff_output,pred_dim_index

def retrieve_patch_index_list(csv_path,masknumber):
    df = load_csv(csv_path)
    df = df[df['Mask Type'] == 'masks_pred']
    patch_index_list = df[(df['Mask Index'] == masknumber) & (df['Patch Index'] == 0)]['Patch Values']
    # Convert each string representation of a list into an actual list
    patch_index_list = patch_index_list.apply(ast.literal_eval).tolist()
    
    # Flatten the list if needed
    patch_index_list = [item for sublist in patch_index_list for item in sublist]
    return patch_index_list


def plot(hi_output,zi_output,hi_zi_diff_output,output_png_path,plot_title):

    # Reshape the lists into 8 frames of 14x14 patches
    hi_output_frames = np.array(hi_output).reshape(8, 14, 14)
    zi_output_frames = np.array(zi_output).reshape(8, 14, 14)
    hi_zi_diff_frames = np.array(hi_zi_diff_output).reshape(8, 14, 14)

    # Calculate min and max for consistent scaling
    hi_min, hi_max = hi_output_frames.min(), hi_output_frames.max()
    zi_min, zi_max = zi_output_frames.min(), zi_output_frames.max()
    diff_min, diff_max = hi_zi_diff_frames.min(), hi_zi_diff_frames.max()
    total_min = min(hi_min,zi_min)
    total_max = max(hi_max,zi_max)
    # Create subplots
    fig, axes = plt.subplots(3, 8, figsize=(24, 12))  # 3 rows (hi, zi, diff) and 8 columns (frames)
    
    # Set the main title for the entire figure
    fig.suptitle(plot_title, fontsize=16, fontweight='bold')
    
    # Plot hi_output
    for i in range(8):
        ax = axes[0, i]
        im = ax.imshow(hi_output_frames[i], cmap='viridis', aspect='equal', vmin=total_min, vmax=total_max)
        ax.set_title(f"Frame {i+1} (hi)")
        ax.axis('off')
        fig.colorbar(im, ax=ax, shrink=0.8)

    # Plot zi_output
    for i in range(8):
        ax = axes[1, i]
        im = ax.imshow(zi_output_frames[i], cmap='viridis', aspect='equal', vmin=total_min, vmax=total_max)
        ax.set_title(f"Frame {i+1} (zi)")
        ax.axis('off')
        fig.colorbar(im, ax=ax, shrink=0.8)

    # Plot hi_zi_diff_output
    for i in range(8):
        ax = axes[2, i]
        im = ax.imshow(hi_zi_diff_frames[i], cmap='viridis', aspect='equal', vmin=diff_min, vmax=diff_max)
        ax.set_title(f"Frame {i+1} (diff)")
        ax.axis('off')
        fig.colorbar(im, ax=ax, shrink=0.8)

    # Adjust layout and save the plot
    plt.tight_layout()
    fig.savefig(output_png_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"plotted figure at {output_png_path}")

def get_clip_index_from_csv(base_dir,epoch_idx,itr_idx,clip_index):
    clip_in_itr_df = pd.read_csv(os.path.join(base_dir,f"masks_csv/epoch_{epoch_idx}","clip_index_list.csv"),
                sep="\t") 
    curr_clip_str = clip_in_itr_df.iloc[itr_idx,1]
    curr_clip_list = ast.literal_eval(curr_clip_str)
    return curr_clip_list[clip_index] # an integer 

def main(base_dir,epoch_itr_list,mask_idx_list,clip_list, pred_dim_index = 0,clip_index = 0):
    for epoch,itr in epoch_itr_list:
        curr_training_clip_index = get_clip_index_from_csv(base_dir,epoch,itr,clip_index)
        plot_title = clip_list[curr_training_clip_index]
        for mask_idx in mask_idx_list:
            hi_path = os.path.join(base_dir,"masks_csv",f"epoch_{epoch}/iter{itr}_hi_mask{mask_idx}.pt")
            zi_path = os.path.join(base_dir,"masks_csv",f"epoch_{epoch}/iter{itr}_zi_mask{mask_idx}.pt")
            csv_path = os.path.join(base_dir,"masks_csv",f"epoch_{epoch}/masks_itr_{itr}.csv")
            patch_index_list = retrieve_patch_index_list(csv_path,mask_idx)

            hi_output,zi_output,hi_zi_diff_output,pred_dim_index = format_hi_zi_to_list(hi_path,zi_path,patch_index_list,pred_dim_index = pred_dim_index, clip_index = clip_index)
            output_png_path = os.path.join(base_dir,"masks_visualization",f"epoch_{epoch}/masks_itr_{itr}/clip{clip_index}_dim{pred_dim_index}_mask{mask_idx}.png")
            if os.path.exists(output_png_path):
                print(f"visualization {output_png_path} exists, skipping")
                pass
            else:
                plot(hi_output,zi_output,hi_zi_diff_output,output_png_path,plot_title)

if __name__ == "__main__":       
    base_dir = "/media/backup_16TB/sean/VJEPA/a6000_output/pretrain_maskfix1_tiny"
    pretraining_csv_path = "/media/backup_16TB/sean/VJEPA/jepa/configs/pretrain_a6000/pretrain_maskfix1/pretrain_maskfix1.csv"
    cliplist_df = pd.read_csv(pretraining_csv_path,sep=' ', header=None)
    cliplist = cliplist_df.iloc[:, 0].tolist()
    epoch_itr_list =[[86,1191],[86,1192],[86,1193],[86,1194],[86,1195],[86,1196],[86,1197]]  
    mask_idx_list = [0]
    pred_dim_index_list = [i for i in range(192)]
    batch_size = 12
    
    for epoch,itr in epoch_itr_list:
        plot_mask_visualization(f"{base_dir}/masks_csv/epoch_{epoch}/masks_itr_{itr}.csv",)
    for clip_index in range(batch_size):
        for pred_dim_index in pred_dim_index_list:
            main(base_dir,epoch_itr_list,mask_idx_list,cliplist,pred_dim_index,clip_index)


