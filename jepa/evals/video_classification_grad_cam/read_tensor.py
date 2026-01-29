#%%
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

def read_tensor(tensor_path,class_id=None):
    if class_id is None:
        class_id = 'not_specified_class'
    parentdir = os.path.dirname(tensor_path)
    # Load the saved Grad-CAM tensors
    data = torch.load(tensor_path)
    print(data.keys())
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
    cam = cam.cpu().numpy() if hasattr(cam, 'cpu') else np.array(cam)
    print('min cam:', cam.min(), 'max cam:', cam.max())
    # cam = np.maximum(cam, 0)  # ReLU

    # Visualize Grad-CAM overlayed on input patches
    input_tensor = data['input']  # shape: [3, 16, 224, 224]
    print('input shape:', input_tensor.shape)

    # Convert input to numpy
    input_np = input_tensor.cpu().numpy() if hasattr(input_tensor, 'cpu') else np.array(input_tensor)

    frames_per_patch = input_np.shape[1] // num_frames  # 16 // 8 = 2
    # Compute frame importance scores
    frame_importance = []

    for i in range(num_frames):
        ###### Grad-CAM magnitude per frame
        # Method 1: Sum of all Grad-CAM values in the frame
        frame_importance_sum = cam[i].sum()
        # Store the importance score (choose one method)
        frame_importance.append(frame_importance_sum)  # or use any other method

        ##### end Grad-CAM magnitude per frame
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

        # Rotate the input patch image 180 degrees
        input_patch_img = np.rot90(input_patch_img, 2)  # Rotate 180 degrees

        # Rotate the Grad-CAM heatmap 180 degrees
        frame_cam_resized = np.rot90(frame_cam_resized, 2)  # Rotate 180 degrees

        # Plot input and Grad-CAM side by side
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(input_patch_img, aspect='equal')  # Ensure consistent aspect ratio
        plt.title(f'Input Patch Mean {i} ({start}-{end-1})')
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(input_patch_img, aspect='equal')  # Ensure consistent aspect ratio
        plt.imshow(frame_cam_resized, cmap='jet', alpha=0.5, aspect='equal')  # Overlay Grad-CAM
        plt.title(f'Grad-CAM Patch {i}')
        plt.axis('off')
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(f'{parentdir}/class_{class_id}_gradcam_patch_{i}.png')
        plt.close()
    print(f'Saved: {parentdir}/class_{class_id}_gradcam_patch_{i}.png')
        # plt.show()


if __name__ == "__main__":
    read_tensor("/media/backup_16TB/sean/VJEPA_results/base/video_classification_frozen/MCICN_alltrainscans_nomeaninten/MCICN_alltrainscans_nomeaninten_run3_gradcam/gradcam_class_1.pt", class_id = 1)
    # # Find the most and least important frames
    # most_important_frame = np.argmax(frame_importance)
    # least_important_frame = np.argmin(frame_importance)

    # print(f"\nMost important frame: {most_important_frame} (score: {frame_importance[most_important_frame]:.4f})")
    # print(f"Least important frame: {least_important_frame} (score: {frame_importance[least_important_frame]:.4f})")


    # # Plot frame importance over time
    # plt.figure(figsize=(10, 6))
    # plt.plot(range(num_frames), frame_importance, 'o-', linewidth=2, markersize=8)
    # plt.xlabel('Frame Index')
    # plt.ylabel('Importance Score')
    # plt.title('Frame Importance Over Time (Grad-CAM)')
    # plt.grid(True, alpha=0.3)
    # plt.xticks(range(num_frames))
    # plt.savefig(f'{parentdir}/class_{class_id}_frame_importance_.png')
    # plt.show()


    # # %%
    # # Select the patch/frame index you believe covers the hippocampus best
    # patch_idx = 0  # Change as needed

    # # activations shape: [num_frames, height, width, channels]
    # act_patch = activations[patch_idx].cpu().numpy() if hasattr(activations, 'cpu') else np.array(activations[patch_idx])  # [14, 14, 768]

    # # Optionally, define a hippocampus ROI (e.g., a box in the 14x14 grid)
    # # Example: ROI is a 4x4 region in the center
    # roi = (slice(3,4), slice(3,10))  # Adjust as needed

    # print(roi)
    # #%%
    # # Compute mean activation in ROI for each channel
    # mean_acts = act_patch[roi[0], roi[1], :].mean(axis=(0,1))  # [768]

    # # Get top-N channels by mean activation in ROI
    # topN = 8
    # top_channels = mean_acts.argsort()[-topN:][::-1]

    # # Plot the activation maps for the top-N channels
    # for idx, ch in enumerate(top_channels):
    #     plt.figure()
    #     plt.imshow(act_patch[:,:,ch], cmap='hot')
    #     plt.title(f'Patch {patch_idx} - Channel {ch} (mean act in ROI: {mean_acts[ch]:.3f})')
    #     plt.colorbar()
    #     plt.show()
    # # %%

    # %%
