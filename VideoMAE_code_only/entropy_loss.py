import torch
import ants
def reshape_tensor_from_recon(video_data):
    recon = video_data.permute(1, 0, 2, 3)  # (channels, temporal, height, width) -> (temporal, channels, height, width)
    recon = recon.reshape(-1,224,224)
    # #resize to (168,168)
    # recon = np.array([resize(slice_c, (112, 112), anti_aliasing=True) for slice_c in recon])
    # recon_ants = ants.from_numpy(recon)
    return recon

def extract_subcortical_entropy_torch(atlas, orig_torch, slice_list, feature_type='entropy'):
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
    
    def calculate_entropy_torch(values, bins=32, sigma=0.01):
        """
        Differentiable entropy computation using soft histogram with Gaussian kernels.
        Stabilized to prevent gradient explosions.
        Args:
            values: 1D tensor of pixel values
            bins: number of histogram bins
            sigma: smoothing parameter for Gaussian kernels (controls differentiability)
        """
        if values.numel() == 0:
            return torch.tensor(0.0, device=values.device)
        
        # Normalize values to [0, 1] range
        min_val, max_val = values.min(), values.max()
        if max_val == min_val:
            return torch.tensor(0.0, device=values.device)  # Constant values -> zero entropy
        
        values_norm = (values - min_val) / (max_val - min_val + 1e-8)
        values_norm = torch.clamp(values_norm, 0.0, 1.0)
        
        # Create bin centers
        bin_width = 1.0 / bins
        bin_centers = torch.linspace(bin_width / 2, 1.0 - bin_width / 2, bins, device=values.device)
        
        # Compute soft assignments using Gaussian kernels (differentiable)
        values_expanded = values_norm.unsqueeze(-1)  # [N, 1]
        bin_centers_expanded = bin_centers.unsqueeze(0)  # [1, bins]
        
        # Gaussian kernel: exp(-(x - mu)^2 / (2 * sigma^2))
        # Using bin_width as sigma gives good balance between smoothness and accuracy
        kernel_sigma = bin_width * sigma * bins  # Scale sigma with bin width
        weights = torch.exp(-((values_expanded - bin_centers_expanded) ** 2) / (2 * kernel_sigma ** 2))
        
        # Sum contributions to each bin and normalize to get probability distribution
        hist = torch.sum(weights, dim=0)  # [bins]
        hist_sum = torch.sum(hist)
        
        # Ensure histogram sum is not zero (shouldn't happen but safety check)
        if hist_sum < 1e-10:
            return torch.tensor(0.0, device=values.device)
        
        hist = hist / hist_sum  # Normalize to sum to 1
        
        # CRITICAL: Clamp probabilities away from 0 and 1 before log to prevent gradient explosion
        # This is the key stabilization - prevents log(0) and extreme gradients from log(small_value)
        eps_min = 1e-6  # Minimum probability (prevents log(0))
        eps_max = 1.0 - 1e-6  # Maximum probability (prevents log(1-epsilon) issues)
        hist_safe = torch.clamp(hist, min=eps_min, max=eps_max)
        
        # Compute Shannon entropy: -sum(p * log2(p))
        # Now safe because hist_safe is clamped away from 0
        entropy = -torch.sum(hist_safe * torch.log2(hist_safe))
        
        # Clamp entropy to reasonable range to prevent extreme values
        # Maximum entropy for uniform distribution: log2(bins)
        # This provides additional gradient stability
        max_entropy = torch.log2(torch.tensor(float(bins), device=values.device, dtype=torch.float32))
        entropy = torch.clamp(entropy, min=0.0, max=max_entropy)
        
        # Final safety check: ensure entropy is finite
        if not torch.isfinite(entropy):
            entropy = torch.tensor(0.0, device=values.device)
        
        return entropy
    
    def calculate_roi_feature(roi_pixels, feature_type='entropy'):
        """
        Calculate a single scalar feature summarizing the ROI.
        
        Volume-based features:
        - 'sum' or 'volume_intensity': Sum of all pixel intensities (intensity-weighted volume)
          * WARNING: If using a fixed atlas template, pixel counts are constant
          * In this case, sum = mean × count, so sum ∝ mean (identical correlation matrices!)
          * Only useful if ROI sizes vary across subjects/slices
        - 'count' or 'volume_count': Number of pixels in ROI (spatial extent)
          * WARNING: With fixed atlas, counts are constant → no variance → useless for correlation
          * Only useful if ROI sizes vary significantly
        
        Distribution-based features:
        - 'entropy': Differentiable entropy (captures distribution shape)
        - 'variance': Variance (captures spatial heterogeneity)
        - 'std': Standard deviation
        - 'mean_std': Weighted combination of mean and std
        - 'iqr': Interquartile range (robust to outliers)
        
        Intensity-based features:
        - 'mean': Mean intensity (normalized volume, loses spatial info)
        """
        if roi_pixels.numel() == 0:
            return torch.tensor(0.0, device=roi_pixels.device, dtype=roi_pixels.dtype)
        
        if feature_type == 'entropy':
            return calculate_entropy_torch(roi_pixels)
        elif feature_type == 'variance':
            return torch.var(roi_pixels)
        elif feature_type == 'std':
            return torch.std(roi_pixels)
        elif feature_type == 'mean_std':
            mean_val = torch.mean(roi_pixels)
            std_val = torch.std(roi_pixels)
            return 2 * mean_val + std_val  # Weight std more
        elif feature_type == 'iqr':
            # Interquartile range (75th - 25th percentile)
            q25 = torch.quantile(roi_pixels, 0.25)
            q75 = torch.quantile(roi_pixels, 0.75)
            return q75 - q25
        elif feature_type in ['sum', 'volume_intensity']:
            # Sum of intensities = intensity-weighted volume
            # This is mean × count, capturing both intensity and spatial extent
            return torch.sum(roi_pixels)
        elif feature_type in ['count', 'volume_count']:
            # Number of pixels (pure spatial extent)
            return torch.tensor(float(roi_pixels.numel()), device=roi_pixels.device, dtype=roi_pixels.dtype)
        else:
            return torch.mean(roi_pixels)  # Default to mean
    
    for roi_id, name in subcortical_labels:
        # Extract the values corresponding to the current region
        mask = (atlas == roi_id)
        
        # Check if ROI exists in these slices
        if mask.sum() == 0:
            # No pixels for this ROI in selected slices
            feature_val = torch.tensor(0.0, device=orig_torch.device, dtype=orig_torch.dtype)
        else:
            # Extract ONLY the pixels within the ROI
            roi_pixels = orig_torch[mask]  # Shape: [num_roi_pixels]
            feature_val = calculate_roi_feature(roi_pixels, feature_type=feature_type)
        
        entropy_data[name] = feature_val

# each sample has 20 roi entropy values
# we have 24 samples in the batch
# calculate the correlation of all combinations of roi across 24 samples. 
# 

    return entropy_data


def calculate_trace_torch(batch_reconstructed_video, batch_indices_list, batch_axis_list, device = None, atlas_list=None, feature_type='entropy', use_trace_sum=True):
    """
    Calculate correlation trace for ROI features.
    
    Args:
        batch_reconstructed_video: List or tensor of reconstructed videos
        batch_indices_list: List or tensor of slice indices for each video
        batch_axis_list: List or tensor of axis values (0, 1, or 2) for each video
        device: Device to run computation on
        atlas_list: List of atlas tensors for each axis
        feature_type: Type of ROI feature to extract. Options:
            Volume-based (NOTE: With fixed atlas, counts are constant, so sum ∝ mean):
            - 'sum' or 'volume_intensity': Sum of intensities (only useful if ROI sizes vary)
            - 'count' or 'volume_count': Number of pixels (useless if counts are constant)
            Distribution-based:
            - 'entropy': Differentiable entropy (recommended, captures distribution)
            - 'variance': Variance (captures spatial heterogeneity)
            - 'std': Standard deviation
            - 'mean_std': Weighted combination (0.3*mean + 0.7*std)
            - 'iqr': Interquartile range (robust to outliers)
            Intensity-based:
            - 'mean': Mean intensity (normalized volume, but loses spatial info)
        use_trace_sum: If True, return trace (sum of diagonal) as a scalar with clamping and normalization.
                       If False, return diagonal elements as a tensor (original behavior).
                       Default: True
    """
    if device is None:
        device = torch.device('cuda')  # Default to GPU 0 if no device is specified
    
    # Handle both tensor and list inputs for indices and axis
    if isinstance(batch_indices_list, torch.Tensor):
        batch_indices_list = [batch_indices_list[i] for i in range(len(batch_reconstructed_video))]
    if isinstance(batch_axis_list, torch.Tensor):
        batch_axis_list = [batch_axis_list[i].item() for i in range(len(batch_reconstructed_video))]
    
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
    num_regions = len(subcortical_labels) / 2

    # Initialize a list to store entropy values for all subjects
    entropy_list = []
    for subject_id, (clip, slice_indices, axis) in enumerate(zip(batch_reconstructed_video, batch_indices_list, batch_axis_list)):
        extract_tensor = reshape_tensor_from_recon(clip)

        if axis == 0:
            entropy_data = extract_subcortical_entropy_torch(atlas_list[0], extract_tensor, slice_indices, feature_type=feature_type)
        elif axis == 1:
            entropy_data = extract_subcortical_entropy_torch(atlas_list[1], extract_tensor, slice_indices, feature_type=feature_type)
        else:
            entropy_data = extract_subcortical_entropy_torch(atlas_list[2], extract_tensor, slice_indices, feature_type=feature_type)
        # Collect entropy values for the current subject
        subject_entropy = [entropy_data[name] for _, name in subcortical_labels]
        entropy_list.append(torch.stack(subject_entropy))  # Stack into a tensor
    # Stack all subjects' entropy tensors into a single tensor
    entropy_tensor = torch.stack(entropy_list, dim=0).to(device).requires_grad_()
    
    # Check for NaN/Inf in entropy tensor before correlation computation
    if not torch.all(torch.isfinite(entropy_tensor)):
        print(f"Warning: NaN/Inf detected in entropy_tensor. Replacing with 0.")
        entropy_tensor = torch.where(torch.isfinite(entropy_tensor), entropy_tensor, torch.zeros_like(entropy_tensor))
    
    # Handle edge case: need at least 2 subjects for correlation
    if entropy_tensor.size(0) < 2:
        num_regions = entropy_tensor.size(1)
        if use_trace_sum:
            return torch.tensor(0.0, device=device, dtype=entropy_tensor.dtype)
        else:
            return torch.zeros(num_regions, device=device, dtype=entropy_tensor.dtype)

    # Compute the correlation matrix on the GPU
    mean = torch.mean(entropy_tensor, dim=0, keepdim=True)
    centered = entropy_tensor - mean
    covariance = (centered.T @ centered) / (entropy_tensor.size(0) - 1)
    
    # Compute variance and ensure it's non-negative (numerical stability)
    variance = torch.diag(covariance)
    variance_safe = torch.clamp(variance, min=0.0)  # Ensure non-negative
    stddev = torch.sqrt(variance_safe)
    
    # Add epsilon to prevent division by very small numbers (gradient explosion)
    # Use larger epsilon for float16 to account for lower precision
    # Check dtype to use appropriate epsilon
    if entropy_tensor.dtype == torch.float16:
        eps = 1e-4  # Larger epsilon for float16 precision
    else:
        eps = 1e-6  # Standard epsilon for float32
    stddev_safe = stddev + eps
    corr_matrix = covariance / (stddev_safe[:, None] * stddev_safe[None, :])
    
    # Clamp correlation values to prevent extreme values that cause gradient issues
    corr_matrix = torch.clamp(corr_matrix, -1.0, 1.0)
    
    # Replace any NaN/Inf with 0 (safety check)
    corr_matrix = torch.where(torch.isfinite(corr_matrix), corr_matrix, torch.zeros_like(corr_matrix))

    # Extract the top-right square of the correlation matrix
    left_regions = [i for i, (_, name) in enumerate(subcortical_labels) if "Left" in name]
    right_regions = [i for i, (_, name) in enumerate(subcortical_labels) if "Right" in name]
    top_right_square = corr_matrix[left_regions, :][:, right_regions]

    # Compute the trace of the top-right square
    if use_trace_sum:
        # Option 1: Use trace (sum of diagonal) with clamping and normalization
        # This returns a single scalar value instead of diagonal elements
        trace = torch.trace(top_right_square)
        trace = torch.clamp(trace, min=0.0, max=float(num_regions))
        trace = trace / float(num_regions)
        
        # Final safety check: ensure trace is finite
        if not torch.isfinite(trace):
            print(f"Warning: NaN/Inf detected in trace. Replacing with 0.")
            trace = torch.tensor(0.0, device=device, dtype=entropy_tensor.dtype)
        
        return trace
    else:
        # Option 2: Return diagonal elements (original behavior)
        trace = torch.diag(top_right_square)
        # Final safety check: ensure trace is finite
        if not torch.all(torch.isfinite(trace)):
            print(f"Warning: NaN/Inf detected in trace. Replacing with 0.")
            trace = torch.where(torch.isfinite(trace), trace, torch.zeros_like(trace))
        return trace