import torch
import ants
def reshape_tensor_from_recon(video_data):
    recon = video_data.permute(1, 0, 2, 3)  # (channels, temporal, height, width) -> (temporal, channels, height, width)
    recon = recon.reshape(-1,224,224)
    # #resize to (168,168)
    # recon = np.array([resize(slice_c, (112, 112), anti_aliasing=True) for slice_c in recon])
    # recon_ants = ants.from_numpy(recon)
    return recon

def extract_subcortical_entropy_torch(atlas, orig_torch, slice_list):
    atlas = atlas[slice_list]
    subcortical_labels = [
        (1, "Cerebral_White_Matter_Left"),
        (2, "Cerebral_Cortex_Left"),
        (3, "Lateral_Ventricle_Left"),
        (4, "Thalamus_Left"),
        (5, "Caudate_Left"),
        (6, "Putamen_Left"),
        (7, "Pallidum_Left"),
        (9, "Hippocampus_Left"),
        (10, "Amygdala_Left"),
        (12, "Cerebral_White_Matter_Right"),
        (13, "Cerebral_Cortex_Right"),
        (14, "Lateral_Ventricle_Right"),
        (15, "Thalamus_Right"),
        (16, "Caudate_Right"),
        (17, "Putamen_Right"),
        (18, "Pallidum_Right"),
        (19, "Hippocampus_Right"),
        (20, "Amygdala_Right"),
        # (11, "Accumbens_Left"),
        # (21, "Accumbens_Right")
        # (8, "Brain-Stem")  # midline single structure
    ]

    entropy_data = {}
    def calculate_mean_torch(values):
        if values.numel() == 0:  # Check if the tensor is empty
            return torch.tensor(0.0, device=values.device)
        return torch.mean(values)

    def calculate_entropy_torch(values):
        if values.numel() == 0:  # Check if the tensor is empty
            return torch.tensor(0.0, device=values.device)

        # Compute a differentiable histogram approximation
        bins = 64
        min_val, max_val = 0.0, 1.0
        values = torch.clamp(values, min=min_val, max=max_val)  # Clamp values to the range
        bin_width = (max_val - min_val) / bins
        bin_centers = torch.linspace(min_val + bin_width / 2, max_val - bin_width / 2, bins, device=values.device)
        
        # Compute soft assignments to bins
        values = values.unsqueeze(-1)  # Add a dimension for broadcasting
        weights = torch.exp(-((values - bin_centers) ** 2) / (2 * (bin_width ** 2)))
        hist = torch.sum(weights, dim=0)  # Sum contributions to each bin
        hist = hist / torch.sum(hist)  # Normalize the histogram to sum to 1
        hist = hist[hist > 0]  # Remove zero probabilities

        # Compute Shannon entropy
        entropy = -torch.sum(hist * torch.log2(hist))
        return entropy
    for roi_id, name in subcortical_labels:
        # Extract the values corresponding to the current region
        region_values = orig_torch * (atlas == roi_id).float()

        # Calculate entropy for the region
        mean = calculate_mean_torch(region_values)

        # Store the entropy value in the dictionary
        entropy_data[f"{name}"] = mean

# each sample has 20 roi entropy values
# we have 24 samples in the batch
# calculate the correlation of all combinations of roi across 24 samples. 
# 

    return entropy_data


def calculate_trace_torch(batch_reconstructed_video, batch_indices_list, batch_axis_list, device = None, atlas_list=None):
    if device is None:
        device = torch.device('cuda')  # Default to GPU 0 if no device is specified
    
    subcortical_labels = [
        (1, "Cerebral_White_Matter_Left"),
        (2, "Cerebral_Cortex_Left"),
        (3, "Lateral_Ventricle_Left"),
        (4, "Thalamus_Left"),
        (5, "Caudate_Left"),
        (6, "Putamen_Left"),
        (7, "Pallidum_Left"),
        (9, "Hippocampus_Left"),
        (10, "Amygdala_Left"),
        (12, "Cerebral_White_Matter_Right"),
        (13, "Cerebral_Cortex_Right"),
        (14, "Lateral_Ventricle_Right"),
        (15, "Thalamus_Right"),
        (16, "Caudate_Right"),
        (17, "Putamen_Right"),
        (18, "Pallidum_Right"),
        (19, "Hippocampus_Right"),
        (20, "Amygdala_Right"),
    ]
    subcortical_labels = sorted(subcortical_labels, key=lambda x: "Right" in x[1])
    # Initialize a tensor to store entropy values for all subjects
    num_subjects = len(batch_reconstructed_video)
    num_regions = len(subcortical_labels)

    # Initialize a list to store entropy values for all subjects
    entropy_list = []
    for subject_id, (clip, slice_indices, axis) in enumerate(zip(batch_reconstructed_video, batch_indices_list, batch_axis_list)):
        extract_tensor = reshape_tensor_from_recon(clip)

        if axis == 0:
            entropy_data = extract_subcortical_entropy_torch(atlas_list[0], extract_tensor, slice_indices)
        elif axis == 1:
            entropy_data = extract_subcortical_entropy_torch(atlas_list[1], extract_tensor, slice_indices)
        else:
            entropy_data = extract_subcortical_entropy_torch(atlas_list[2], extract_tensor, slice_indices)
        # Collect entropy values for the current subject
        subject_entropy = [entropy_data[name] for _, name in subcortical_labels]
        entropy_list.append(torch.stack(subject_entropy))  # Stack into a tensor
    # Stack all subjects' entropy tensors into a single tensor
    entropy_tensor = torch.stack(entropy_list, dim=0).to(device).requires_grad_()

    # Compute the correlation matrix on the GPU
    mean = torch.mean(entropy_tensor, dim=0, keepdim=True)
    centered = entropy_tensor - mean
    covariance = (centered.T @ centered) / (entropy_tensor.size(0) - 1)
    stddev = torch.sqrt(torch.diag(covariance))
    corr_matrix = covariance / (stddev[:, None] * stddev[None, :])

    # Extract the top-right square of the correlation matrix
    left_regions = [i for i, (_, name) in enumerate(subcortical_labels) if "Left" in name]
    right_regions = [i for i, (_, name) in enumerate(subcortical_labels) if "Right" in name]
    top_right_square = corr_matrix[left_regions, :][:, right_regions]

    # Compute the trace of the top-right square
    trace = torch.trace(top_right_square) 
    trace = torch.clamp(trace, min=0.0, max=num_regions)
    trace = trace / num_regions
    return trace


