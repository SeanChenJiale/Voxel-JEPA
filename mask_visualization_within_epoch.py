import matplotlib.pyplot as plt
import numpy as np
import ast
import pandas as pd
import os




def plot_mask_visualization(csv_path):
    def create_directory(mask_dir):
        """Create a directory for storing the visualizations based on the mask directory.
        outputs a directory path that is the parent directory of the mask directory with a new name."""

        root_output_dir = os.path.dirname(mask_dir) ## assumes the output directory is the parent directory of the mask directory
        output_dir_name = os.path.basename(mask_dir) + "_visualization"
        output_base_dir = os.path.join(root_output_dir, output_dir_name)

        if not os.path.exists(mask_dir):
            os.makedirs(mask_dir, exist_ok=True)
            # print(f"Created directory: {mask_dir}")
        else:
            pass
            # print(f"Directory already exists: {mask_dir}")
        return output_base_dir

    def access_epochs(mask_dir,epoch_list=None):
        """if epoch list is None, return all paths to mask files within epochs"""
        """if epoch list is provided, return paths to mask files within those epochs"""
        mask_itr_list = [string for string in os.listdir(mask_dir) if "masks_itr" in string]
        if epoch_list is None:
            return mask_itr_list
        else:
            return [mask_itr for mask_itr in mask_itr_list if any(f"epoch_{epoch}" in mask_itr for epoch in epoch_list)]
        
    """Plot the mask visualization from a CSV file and save it to the output path."""
    def get_epoch_number(csv_path):
        """Extract the epoch number from the CSV file path."""
        epoch_str = csv_path.split("epoch_")[-1].split("/")[0]
        return int(epoch_str)
    def get_itr_number(csv_path):
        """Extract the iteration number from the CSV file path."""
        itr_str = csv_path.split("masks_itr_")[-1].split(".csv")[0]
        return int(itr_str)
    def create_epoch_directory(csv_path):
        """Create a directory for storing the visualizations based on the CSV file path."""
        root_output_dir = os.path.dirname(csv_path)
        
        #replace masks_csv with masks_visualization
        output_base_dir = root_output_dir.replace("masks_csv", "masks_visualization")

        # print(f"Creating directory: {output_base_dir}")
        if not os.path.exists(output_base_dir):
            os.makedirs(output_base_dir, exist_ok=True)
            # print(f"Created directory: {output_base_dir}")
        else:
            pass
            # print(f"Directory already exists: {output_base_dir}")
        return output_base_dir

    def create_output_directory(csv_path):
        itr = get_itr_number(csv_path)
        epoch = get_epoch_number(csv_path)
        output_base_dir = create_epoch_directory(csv_path)
        """Create a directory for storing the visualizations based on the CSV file path."""
        output_base_dir = os.path.join(output_base_dir, f"masks_itr_{itr}")
        # print(f"Creating directory: {output_base_dir}")
        if not os.path.exists(output_base_dir):
            os.makedirs(output_base_dir, exist_ok=True)
            # print(f"Created directory: {output_base_dir}")
        else:
            pass
            # print(f"Directory already exists: {output_base_dir}")
        return output_base_dir

    output_base_dir = create_output_directory(csv_path)
    df = pd.read_csv(csv_path)
    
    # Convert the `Patch Values` column from string to list
    df['Patch Values'] = df['Patch Values'].apply(ast.literal_eval)

    # Filter df to include only "masks_enc"
    df = df[df['Mask Type'].isin(["masks_pred"])]

    # print(df.info())
    # Define the grid dimensions
    grid_height = 14
    grid_width = 14
    num_frames = 8
    
    # Define the full range of patch indices (0 to 1567 inclusive)
    full_patch_range = set(range(0,(14*14+1)))  # 14x14 grid for 8 frames

    # Group rows by the combination of `Mask Type` and `Patch Index`
    grouped = df.groupby(["Mask Type", "Mask Index"])



    # Iterate over the groups
    for (mask_type, patch_index), group in grouped:
        if os.path.exists(f"{output_base_dir}/mask_{patch_index}.png"):
            print(f"{output_base_dir}/mask_{patch_index}.png exists, skipping")
            pass
        else:
            # print(f"Group for Mask Type = {mask_type}, Patch Index = {patch_index}:")
            # print(group)

            # Access the `Patch Values` column for this group
            patch_values_list = group["Patch Values"].tolist()
            
            # Flatten the list of lists into a single set of all patch values in the group
            all_patch_values = set([value for sublist in patch_values_list for value in sublist])
            
            # Calculate the missing indices (union of all missing indices)
            missing_indices = full_patch_range - all_patch_values
            # print(f"Missing indices for this group: {sorted(missing_indices)}")

            # Create a 14x14 grid for each frame
            frame = 0
            grid = np.zeros((grid_height, grid_width))  # Initialize grid with white (1)
            beginning_patch = frame * grid_width * grid_height
            ending_patch = (frame + 1) * grid_width * grid_height
            # print(f"this is the beginning patch index {beginning_patch}")
            # print(f"this is the ending patch index {ending_patch}")
            # Get the patch values for the current frame
            patch_values = missing_indices
            
            filtered_patch_values = [patch for patch in patch_values if beginning_patch <= patch < ending_patch]
            filtered_patch_values = [patch - beginning_patch for patch in filtered_patch_values]

            for patch in filtered_patch_values:
                row = patch // grid_width
                col = patch % grid_width
                grid[row, col] = 1
            # Plot the grid
            plt.figure(figsize=(5, 5))
            plt.title(f"{mask_type} - Clip {patch_index + 1}")
            plt.imshow(grid, cmap="gray", vmin=0, vmax=1)

            # Show the axes
            plt.xlabel("Columns")
            plt.ylabel("Rows")
            plt.xticks(range(0, grid_width, 2))  # Optional: Customize x-axis ticks
            plt.yticks(range(0, grid_height, 2))  # Optional: Customize y-axis ticks

            # Add a legend
            if mask_type == "masks_enc":
                blackarea_legend = 'Masked'
                whitearea_legend = 'Unmasked'
            else:
                blackarea_legend = 'Masked'
                whitearea_legend = 'Unmasked'

            greyarea_legend = 'Missing Patches'

            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='black', edgecolor='black', label=blackarea_legend),
                Patch(facecolor='white', edgecolor='black', label=whitearea_legend)
            ]
            plt.legend(handles=legend_elements, loc='upper right', title="Legend")

            # Save the grid as a PNG file
            
            output_file = f"{output_base_dir}/mask_{patch_index}.png"
            plt.savefig(output_file, bbox_inches="tight")
            plt.close()
    return
if __name__ == "__main__":
    basedir = "/media/backup_16TB/sean/VJEPA/a6000_output/pretrain_maskfix1_mri"
    epoch_itr_list = [[86,1191],[86,1192],[86,1193],[86,1194],[86,1195],[86,1196],[86,1197]] 
    for epoch,itr in epoch_itr_list:
        plot_mask_visualization(f"/media/backup_16TB/sean/VJEPA/a6000_output/debug/masks_csv/epoch_{epoch}/masks_itr_{itr}.csv",)
