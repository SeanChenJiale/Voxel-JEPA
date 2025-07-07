import torch 
import pandas as pd
import re
import ast
import os
import matplotlib.pyplot as plt
import numpy as np
from mask_visualization_within_epoch import plot_mask_visualization
import pdb
import csv
import tqdm

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
    hi_zi_add_output = [0 for _ in range(1568)]

    hi_clip,zi_clip = select_clip(hi_0,zi_0,clip_index)
    pred_patch_value_hi_list = hi_clip[pred_dim_index].tolist()
    pred_patch_value_zi_list = zi_clip[pred_dim_index].tolist()

    for value_hi, value_zi, patch_index in zip(pred_patch_value_hi_list,pred_patch_value_zi_list,patch_index_list):
        hi_output[patch_index] = value_hi
        zi_output[patch_index] = value_zi
        hi_zi_diff_output[patch_index] = abs(value_hi-value_zi)
        hi_zi_add_output[patch_index] = (value_hi+value_zi)
    return hi_output,zi_output,hi_zi_diff_output,hi_zi_add_output,pred_dim_index

def retrieve_patch_index_list(csv_path,masknumber):
    df = load_csv(csv_path)
    df = df[df['Mask Type'] == 'masks_pred']
    patch_index_list = df[(df['Mask Index'] == masknumber) & (df['Patch Index'] == 0)]['Patch Values']
    # Convert each string representation of a list into an actual list
    patch_index_list = patch_index_list.apply(ast.literal_eval).tolist()
    
    # Flatten the list if needed
    patch_index_list = [item for sublist in patch_index_list for item in sublist]
    return patch_index_list

def check_create_dir(dir_to_create):
    if not os.path.exists(dir_to_create):
        os.makedirs(dir_to_create, exist_ok=True)

def plot(hi_output,zi_output,hi_zi_diff_output,hi_zi_add_output,output_png_path,plot_title):

    # Reshape the lists into 8 frames of 14x14 patches
    hi_output_frames = np.array(hi_output).reshape(8, 14, 14)
    zi_output_frames = np.array(zi_output).reshape(8, 14, 14)
    hi_zi_diff_frames = np.array(hi_zi_diff_output).reshape(8, 14, 14)
    hi_zi_add_frames = np.array(hi_zi_add_output).reshape(8, 14, 14)

    # Calculate min and max for consistent scaling
    hi_min, hi_max = hi_output_frames.min(), hi_output_frames.max()
    zi_min, zi_max = zi_output_frames.min(), zi_output_frames.max()
    diff_min, diff_max = hi_zi_diff_frames.min(), hi_zi_diff_frames.max()
    add_min, add_max = hi_zi_add_frames.min(), hi_zi_add_frames.max()

    total_min = min(hi_min,zi_min)
    total_max = max(hi_max,zi_max)
    # Create subplots
    fig, axes = plt.subplots(4, 8, figsize=(24, 12))  # 3 rows (hi, zi, diff) and 8 columns (frames)
    
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

    # Plot hi_zi_add_output
    for i in range(8):
        ax = axes[3, i]
        im = ax.imshow(hi_zi_add_frames[i], cmap='viridis', aspect='equal', vmin=add_min, vmax=add_max)
        ax.set_title(f"Frame {i+1} (add)")
        ax.axis('off')
        fig.colorbar(im, ax=ax, shrink=0.8)

    # Adjust layout and save the plot
    plt.tight_layout()
    fig.savefig(output_png_path, dpi=50, bbox_inches='tight')
    plt.close(fig)
    print(f"plotted figure at {output_png_path}")

def get_clip_index_from_csv(base_dir,epoch_idx,itr_idx,clip_index,masks_csv_str):
    clip_in_itr_df = pd.read_csv(os.path.join(base_dir,f"{masks_csv_str}/epoch_{epoch_idx}","clip_index_list.csv"),
                sep="\t") 
    curr_clip_str = clip_in_itr_df.iloc[itr_idx,1]
    curr_clip_list = ast.literal_eval(curr_clip_str)
    return curr_clip_list[clip_index] # an integer 

def get_epoch_itr_from_clip_trace_idx(base_dir,epoch_list,clip_trace_idx,masks_csv_str):
    epoch_idx_tracer_list = []
    for epoch in epoch_list:
        
        file_path = os.path.join(base_dir, f"{masks_csv_str}/epoch_{epoch}", "clip_index_list.csv")
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue

        clip_in_itr_df = pd.read_csv(file_path, sep="\t")
        # print("Columns in DataFrame:", clip_in_itr_df.columns)

        if ' clip_index_list' not in clip_in_itr_df.columns:
            print("Column 'clip_index_list' not found in DataFrame.")
            return

        # Convert the 'clip_index_list' column to actual lists
        clip_in_itr_df[' clip_index_list'] = clip_in_itr_df[' clip_index_list'].apply(ast.literal_eval)

        # Filter rows where clip_trace_idx matches exactly in the clip_index_list
        result_itr = clip_in_itr_df.loc[clip_in_itr_df[' clip_index_list'].apply(lambda x: clip_trace_idx in x), 'itr']
        
        
        # Extract the value directly
        if not result_itr.empty:
            itr_value = result_itr.iloc[0]  # Get the first value directly
            # print(itr_value)
            clip_idx_in_itr = clip_in_itr_df.iloc[itr_value][' clip_index_list'].index(clip_trace_idx)
            epoch_idx_tracer_list.append([epoch,itr_value,clip_idx_in_itr])
        else:
            print("No match found.")

    return epoch_idx_tracer_list

def main(base_dir,epoch,itr,mask_idx_list,clip_list, mask_csv_and_vis_list=["masks_csv","masks_visualization"],pred_dim_index_list = [0],clip_index = 0,clip_trace_folder = ""):
    masks_csv_str = mask_csv_and_vis_list[0]
    masks_vis_str = mask_csv_and_vis_list[1]
    curr_training_clip_index = get_clip_index_from_csv(base_dir,epoch,itr,clip_index,masks_csv_str)
    total_loss_per_dimension_list = [0 for _ in range(192)]
    for mask_idx in mask_idx_list:
        for pred_dim_index in pred_dim_index_list:
            # plot_title = f"epoch{epoch}_dim{pred_dim_index}_itr{itr}_{clip_list[curr_training_clip_index]}"
            # output_png_path = os.path.join(base_dir,masks_vis_str,clip_trace_folder,f"dim{pred_dim_index}",f"epoch_{epoch}_masks_itr_{itr}_clip{clip_index}_dim{pred_dim_index}_mask{mask_idx}.png")
            # check_create_dir(os.path.join(base_dir,masks_vis_str,clip_trace_folder,f"dim{pred_dim_index}"))
            hi_path = os.path.join(base_dir,masks_csv_str,f"epoch_{epoch}/iter{itr}_hi_mask{mask_idx}.pt")
            zi_path = os.path.join(base_dir,masks_csv_str,f"epoch_{epoch}/iter{itr}_zi_mask{mask_idx}.pt")
            csv_path = os.path.join(base_dir,masks_csv_str,f"epoch_{epoch}/masks_itr_{itr}.csv")
            patch_index_list = retrieve_patch_index_list(csv_path,mask_idx)

            hi_output,zi_output,hi_zi_diff_output,hi_zi_add_output,pred_dim_index = format_hi_zi_to_list(hi_path,zi_path,patch_index_list,pred_dim_index = pred_dim_index, clip_index = clip_index)
            total_loss_per_dimension_list[pred_dim_index] += sum(hi_zi_diff_output) / (len(mask_idx_list) *len(patch_index_list))
    
    
    return total_loss_per_dimension_list
def plot_loss_visualizations(csv_path, output_dir, clip_trace_str = "", loss_columns_prefix="dimension", num_dimensions=192, heatmap_filename="heatmap.png", lineplot_filename="lineplot.png",plot_line=False):
    """
    Plots a heatmap and line plot for loss values across dimensions.

    Parameters:
        csv_path (str): Path to the CSV file containing the loss data.
        output_dir (str): Directory where the plots will be saved.
        loss_columns_prefix (str): Prefix for the loss columns in the CSV file. Default is "dimension".
        num_dimensions (int): Number of dimensions (loss columns) to plot. Default is 192.
        heatmap_filename (str): Filename for the heatmap plot. Default is "heatmap.png".
        lineplot_filename (str): Filename for the line plot. Default is "lineplot.png".
    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    # Load the CSV file
    df = pd.read_csv(csv_path)

    # Generate loss column names
    loss_columns = [f"{loss_columns_prefix}{i}" for i in range(num_dimensions)]
    loss_data = df[loss_columns]

    # Plot a heatmap
    plt.figure(figsize=(25, 25))
    plt.rcParams['font.size'] = 9
    sns.heatmap(loss_data.T, cmap="viridis", cbar=True, xticklabels=False, yticklabels=loss_columns)
    plt.title(f"Heatmap of Losses Across Dimensions, Clip: {clip_trace_str}")
    plt.xlabel("Iterations")
    plt.ylabel("Dimensions")
    heatmap_path = os.path.join(output_dir, heatmap_filename)
    plt.savefig(heatmap_path)
    plt.close()

    if plot_line:
        # Plot a line plot for trends
        plt.figure(figsize=(25, 25))
        plt.rcParams['font.size'] = 9
        for col in loss_columns:
            plt.plot(df[col], alpha=0.5, label=col)
        plt.title("Loss Trends Across Dimensions")
        plt.xlabel("Iterations")
        plt.ylabel("Loss Value")
        plt.legend(loc="upper right", ncol=4, fontsize="small")
        lineplot_path = os.path.join(output_dir, lineplot_filename)
        plt.savefig(lineplot_path)
        plt.close()
        print(f"Plots saved: Heatmap -> {heatmap_path}, Line Plot -> {lineplot_path}")
    
    print(f"Plots saved: Heatmap -> {heatmap_path}")
   

if __name__ == "__main__":       
    # base_dir = "/media/backup_16TB/sean/VJEPA/a6000_output/fix13_axial"
    # pretraining_csv_path = "/media/backup_16TB/sean/VJEPA/jepa/configs/pretrain_a6000/pretrain_maskfix1/pretrain2_axial.csv"
    # cliplist_df = pd.read_csv(pretraining_csv_path,sep=' ', header=None)
    # cliplist = cliplist_df.iloc[:, 0].tolist()
    # epoch_list = [i for i in range(0,101)]
    # mask_csv_and_vis_list = ["masks_csv","clip_trace_visualization"]
    # total_multiblock_num = 4
    # mask_idx_list = [ _ for _ in range(total_multiblock_num)]
    # pred_dim_index_list =  [ _ for _ in range(192)]
    # # clip_index_list = [ _ for _ in range(12)]
    # clip_trace_str = "/media/backup_16TB/sean/Monai/BrainSlice48_224_224/NACC_T1_Volume_mri/1.2.840.113619.2.408.5282380.4561270.27867.1528128569.787/Axial.mp4"
    # clip_trace_folder = "NACC_T1_Volume_mri_1.2.840.113619.2.408.5282380.4561270.27867.1528128569.787"

    base_dir = "/media/backup_16TB/sean/VJEPA/a6000_output/pretrain_maskfix1_tiny"
    pretraining_csv_path = "/media/backup_16TB/sean/VJEPA/jepa/configs/pretrain_a6000/pretrain_maskfix1/pretrain3_zoom.csv"
    cliplist_df = pd.read_csv(pretraining_csv_path,sep=' ', header=None)
    cliplist = cliplist_df.iloc[:, 0].tolist()
    epoch_list = [i for i in range(0,101)]
    mask_csv_and_vis_list = ["masks_csv","clip_trace_visualization_fix14"]
    total_multiblock_num = 2
    mask_idx_list = [ _ for _ in range(total_multiblock_num)]
    pred_dim_index_list = [ _ for _ in range(192)]
    # clip_index_list = [ _ for _ in range(12)]
    clip_trace_str = "/media/backup_16TB/sean/Monai/BrainSlice48_zoom/ADNI_MP_RAGE/2008-02-23_08_55_00.0_ADNI_12M5_TS_2_20080223085330_2/Sagittal.mp4" # change this
    clip_trace_folder = "2008-02-23_08_55_00.0_ADNI_12M5_TS_2_20080223085330_2_Sagittal.mp4" #feel free to change folder name
    output_folder = os.path.join(base_dir,mask_csv_and_vis_list[1],clip_trace_folder)
    check_create_dir(output_folder)
    # Assuming cliplist_df is your DataFrame and clip_trace_str is the string to match
    matched_row = cliplist_df.loc[cliplist_df.iloc[:, 0] == clip_trace_str]

    if not matched_row.empty:
        print("Exact match found:")
        print(matched_row.index.tolist()[0])
        clip_trace_idx = matched_row.index.tolist()[0]
    else:
        print("No exact match found.")
        
    epoch_itr_clip_idx_list = get_epoch_itr_from_clip_trace_idx(base_dir,epoch_list,clip_trace_idx,masks_csv_str = mask_csv_and_vis_list[0])
    print(f"this is  {epoch_itr_clip_idx_list}")
    csv_path = os.path.join(base_dir,mask_csv_and_vis_list[1],f"{clip_trace_folder}.csv")
    if not os.path.exists(csv_path):
        with open((csv_path),"w") as fname:
            writer = csv.writer(fname)
            writer.writerow([f"dimension{i}" for i in range(192)])
            for epoch,itr,clip_index in tqdm.tqdm(epoch_itr_clip_idx_list):
                # plot_mask_visualization(f"{base_dir}/{mask_csv_and_vis_list[0]}/epoch_{epoch}/masks_itr_{itr}.csv",mask_csv_and_vis_list=mask_csv_and_vis_list)
                loss_for_epoch = main(base_dir,
                    epoch,
                    itr,
                    mask_idx_list,
                    cliplist,
                    pred_dim_index_list=pred_dim_index_list,
                    clip_index=clip_index,
                    mask_csv_and_vis_list=mask_csv_and_vis_list,
                    clip_trace_folder=clip_trace_folder)
                writer.writerow(loss_for_epoch)
    else:
        print(f"{csv_path} exists, skipping creation of csv.")
    # Example usage
    
    
    plot_loss_visualizations(csv_path, output_folder, clip_trace_str)
