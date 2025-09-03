#%%
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

tensor_path = '/media/backup_16TB/sean/VJEPA/a6000_output/vit_base/K400/video_classification_frozen/vit_base_mriload_k40045_allmodality_coronalonly_AD_sample/vit_base_mriload_k40045_allmodality_coronalonly_AD_sample_run2_gradcam/gradcam_sample_0_0.pt'
parentdir = os.path.dirname(tensor_path)
# Load the saved Grad-CAM tensors
data = torch.load(tensor_path)
print(data.keys())
print(data['input'].shape)

#%%
activations = data['activations']  # shape: [1568, 768]
gradients = data['gradients']      # shape: [1568, 768]
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
    plt.savefig(f'{parentdir}/gradcam_patch_{i}.png')
    plt.show()
# %%
# Select the patch/frame index you believe covers the hippocampus best
patch_idx = 0  # Change as needed

# activations shape: [num_frames, height, width, channels]
act_patch = activations[patch_idx].cpu().numpy() if hasattr(activations, 'cpu') else np.array(activations[patch_idx])  # [14, 14, 768]

# Optionally, define a hippocampus ROI (e.g., a box in the 14x14 grid)
# Example: ROI is a 4x4 region in the center
roi = (slice(3,4), slice(3,10))  # Adjust as needed

print(roi)
#%%
# Compute mean activation in ROI for each channel
mean_acts = act_patch[roi[0], roi[1], :].mean(axis=(0,1))  # [768]

# Get top-N channels by mean activation in ROI
topN = 8
top_channels = mean_acts.argsort()[-topN:][::-1]

# Plot the activation maps for the top-N channels
for idx, ch in enumerate(top_channels):
    plt.figure()
    plt.imshow(act_patch[:,:,ch], cmap='hot')
    plt.title(f'Patch {patch_idx} - Channel {ch} (mean act in ROI: {mean_acts[ch]:.3f})')
    plt.colorbar()
    plt.show()
# %%
