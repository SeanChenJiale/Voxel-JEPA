#%%
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

def read_tensor(tensor_path_1, tensor_path_2, class_id=None):
    if class_id is None:
        class_id = 'not_specified_class'
    parentdir = os.path.dirname(tensor_path_1)

    # Load the saved Grad-CAM tensors for both paths
    data_1 = torch.load(tensor_path_1)
    data_2 = torch.load(tensor_path_2)

    # Extract Grad-CAM data
    activations_1 = data_1['avg_activations']  # shape: [1568, 768]
    gradients_1 = data_1['avg_gradients']      # shape: [1568, 768]
    activations_2 = data_2['avg_activations']  # shape: [1568, 768]
    gradients_2 = data_2['avg_gradients']      # shape: [1568, 768]

    # Set parameters
    num_frames = 8
    height = 14
    width = 14
    channels = 768

    # Reshape activations and gradients
    activations_1 = activations_1.view(num_frames, height, width, channels)
    gradients_1 = gradients_1.view(num_frames, height, width, channels)
    activations_2 = activations_2.view(num_frames, height, width, channels)
    gradients_2 = gradients_2.view(num_frames, height, width, channels)

    # Compute Grad-CAM heatmaps
    weights_1 = gradients_1.mean(dim=(0, 1, 2))  # shape: [channels]
    cam_1 = (weights_1 * activations_1).sum(dim=3)  # shape: [num_frames, height, width]
    cam_1 = cam_1.cpu().numpy() if hasattr(cam_1, 'cpu') else np.array(cam_1)

    weights_2 = gradients_2.mean(dim=(0, 1, 2))  # shape: [channels]
    cam_2 = (weights_2 * activations_2).sum(dim=3)  # shape: [num_frames, height, width]
    cam_2 = cam_2.cpu().numpy() if hasattr(cam_2, 'cpu') else np.array(cam_2)

    # Input tensor
    input_tensor = data_1['input']  # shape: [3, 16, 224, 224]
    input_np = input_tensor.cpu().numpy() if hasattr(input_tensor, 'cpu') else np.array(input_tensor)

    frames_per_patch = input_np.shape[1] // num_frames  # 16 // 8 = 2

    for i in range(num_frames):
        # Grad-CAM heatmap for path 1
        frame_cam_1 = cam_1[i].astype(np.float32)
        frame_cam_1 -= frame_cam_1.min()
        if frame_cam_1.max() > 0:
            frame_cam_1 /= frame_cam_1.max()
        frame_cam_resized_1 = cv2.resize(frame_cam_1, (224, 224))
        frame_cam_resized_1 = np.rot90(frame_cam_resized_1, 2)  # Rotate 180 degrees

        # Grad-CAM heatmap for path 2
        frame_cam_2 = cam_2[i].astype(np.float32)
        frame_cam_2 -= frame_cam_2.min()
        if frame_cam_2.max() > 0:
            frame_cam_2 /= frame_cam_2.max()
        frame_cam_resized_2 = cv2.resize(frame_cam_2, (224, 224))
        frame_cam_resized_2 = np.rot90(frame_cam_resized_2, 2)  # Rotate 180 degrees

        # Input patch
        start = i * frames_per_patch
        end = (i + 1) * frames_per_patch
        input_patch = input_np[:, start:end, :, :]  # shape: [3, 2, 224, 224]
        input_patch_mean = input_patch.mean(axis=1)  # shape: [3, 224, 224]
        input_patch_img = np.transpose(input_patch_mean, (1, 2, 0))  # [224, 224, 3]
        input_patch_img = (input_patch_img - input_patch_img.min()) / (input_patch_img.max() - input_patch_img.min() + 1e-8)
        input_patch_img = np.rot90(input_patch_img, 2)  # Rotate 180 degrees

        # Plot input and Grad-CAMs side by side
        plt.figure(figsize=(12, 3))  # Adjust figure size for 3 subplots
        plt.subplot(1, 3, 1)
        plt.imshow(input_patch_img, aspect='equal')
        plt.title(f'Input {i}')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(input_patch_img, aspect='equal')
        plt.imshow(frame_cam_resized_1, cmap='jet', alpha=0.5, aspect='equal')
        plt.title(f'Grad-CAM No NAI-SSL')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(input_patch_img, aspect='equal')
        plt.imshow(frame_cam_resized_2, cmap='jet', alpha=0.5, aspect='equal')
        plt.title(f'Grad-CAM with NAI-SSL')
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(f'{parentdir}/class_{class_id}_gradcam_patch_{i}.png')
        print(f'Saved: {parentdir}/class_{class_id}_gradcam_patch_{i}.png')

if __name__ == "__main__":
    path1 = "/media/backup_16TB/sean/VJEPA_results/base/video_classification_frozen/MCICN_alltrainscans_nomeaninten/MCICN_alltrainscans_nomeaninten_gradcam/gradcam_class_0.pt"
    path2 = "/media/backup_16TB/sean/VJEPA_results/base/video_classification_frozen/MCICN/MCICN_0_05_std_gradcam/gradcam_class_0.pt"
    read_tensor(path1, path2,  class_id = 0)