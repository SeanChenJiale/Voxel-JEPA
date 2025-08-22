## new functionality
import csv
import os

def save_masks_to_csv(masks_enc, masks_pred, filename="masks.csv", logger=None):
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        
        # Write header
        writer.writerow(["Mask Type", "Mask Index", "Patch Index", "Patch Values"])
        
        # Save masks_enc
        for i, m in enumerate(masks_enc):
            for j in range(len(m)):
                patch_values = m[j].cpu().numpy().flatten().tolist()  # Convert tensor to list
                writer.writerow(["masks_enc", i, j, patch_values])
        
        # Save masks_pred
        for i, m in enumerate(masks_pred):
            for j in range(len(m)):
                patch_values = m[j].cpu().numpy().flatten().tolist()  # Convert tensor to list
                writer.writerow(["masks_pred", i, j, patch_values])
    # try:
    #     logger.info(f'Saved csv to {filename}')
    # except:
    #     print(f'Csv for {filename} could not be saved')