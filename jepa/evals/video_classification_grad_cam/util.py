#%%
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

def save_gradcam(tensor_path = None,class_id=None):
    if class_id is None:
        class_id = 'not_specified_class'
    if tensor_path is not None:
        
        parentdir = os.path.dirname(tensor_path)
        # Load the saved Grad-CAM tensors
        data = torch.load(tensor_path)
        # print(data.keys())
        # print(data['input'].shape)

        #%%
        activations = data['avg_activations']  # shape: [1568, 768]
        gradients = data['avg_gradients']      # shape: [1568, 768]
        print('activations shape:', activations.shape)
        print('gradients shape:', gradients.shape)

        # Set your parameters (adjust as needed)
        num_frames = 8
        height = 14
        width = 14
        channels = 768

        # Reshape to [num_frames, height, width, channels]
        activations = activations.view(num_frames, height, width, channels)
        gradients = gradients.view(num_frames, height, width, channels)

        # Average gradients over spatial dimensions (frames, height, width)
        weights = gradients.mean(dim=(0, 1, 2))  # shape: [channels]

        # Weighted sum of activations
        cam = (weights * activations).sum(dim=3)  # shape: [num_frames, height, width]

        # Take mean over frames if you want a single heatmap

        # Take mean over frames if you want a single heatmap
        cam = cam.cpu().numpy() if hasattr(cam, 'cpu') else np.array(cam)
        cam = np.maximum(cam, 0)  # ReLU

        # Visualize Grad-CAM overlayed on input patches
        input_tensor = data['input']  # shape: [3, 16, 224, 224]
        print('input shape:', input_tensor.shape)

        # Convert input to numpy
        input_np = input_tensor.cpu().numpy() if hasattr(input_tensor, 'cpu') else np.array(input_tensor)

        frames_per_patch = input_np.shape[1] // num_frames  # 16 // 8 = 2

        for i in range(num_frames):
            # Grad-CAM heatmap
            frame_cam = cam[i].astype(np.float32)
            frame_cam -= frame_cam.min()
            if frame_cam.max() > 0:
                frame_cam /= frame_cam.max()
            frame_cam_resized = cv2.resize(frame_cam, (224, 224))

            # Corresponding input frames (mean over the patch)
            start = i * frames_per_patch
            end = (i + 1) * frames_per_patch
            input_patch = input_np[:, start:end, :, :]  # shape: [3, 2, 224, 224]
            input_patch_mean = input_patch.mean(axis=1)  # shape: [3, 224, 224]
            input_patch_img = np.transpose(input_patch_mean, (1, 2, 0))  # [224, 224, 3]
            input_patch_img = (input_patch_img - input_patch_img.min()) / (input_patch_img.max() - input_patch_img.min() + 1e-8)

            # Plot input and Grad-CAM side by side
            plt.figure(figsize=(8, 4))
            plt.subplot(1, 2, 1)
            plt.imshow(input_patch_img)
            plt.title(f'Input Patch Mean {i} ({start}-{end-1})')
            plt.axis('off')
            plt.subplot(1, 2, 2)
            plt.imshow(input_patch_img)
            plt.imshow(frame_cam_resized, cmap='jet', alpha=0.5)
            plt.title(f'Grad-CAM Patch {i}')
            plt.axis('off')
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.tight_layout()
            plt.savefig(f'{parentdir}/gradcam_patch_{i}_{class_id}.png')
            # plt.show()
        print(f"\n\n Saved gradcam output to {tensor_path} \n\n")
    else:
        print("\n\n\n Error in locating Tensor Path \n\n\n")