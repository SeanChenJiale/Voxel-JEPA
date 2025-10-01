import cv2  # OpenCV for image processing
import numpy as np  # NumPy for numerical operations

class GradCam:
    def __init__(self, model, target):
        self.model = model.eval()  # Set the model to evaluation mode
        self.feature = None  # [1, 768, 14, 14] To store the features from the target layer
        self.gradient = None  # [1, 768, 14, 14] To store the gradients from the target layer 
        self.handlers = []  # List to keep track of hooks
        self.target = target  # LayerNorm((768,), eps=1e-06, elementwise_affine=True) Target layer for Grad-CAM
        self._get_hook()  # Register hooks to the target layer

    # Hook to get features from the forward pass
    def _get_features_hook(self, module, input, output):
        print(f"Feature shape: {output.shape}")  # Debugging statement
        self.feature = self.reshape_transform(output)  # Store and reshape the output features

    # Hook to get gradients from the backward pass
    def _get_grads_hook(self, module, input_grad, output_grad):
        self.gradient = self.reshape_transform(output_grad)  # Store and reshape the output gradients

        def _store_grad(grad):
            self.gradient = self.reshape_transform(grad)  # Store gradients for later use

        output_grad.register_hook(_store_grad)  # Register hook to store gradients

    # Register forward hooks to the target layer
    def _get_hook(self):
        self.target.register_forward_hook(self._get_features_hook)
        self.target.register_forward_hook(self._get_grads_hook)

    # Function to reshape the tensor for visualization
    def reshape_transform(self, tensor, num_frames=8, height=14, width=14):
        """
        Reshape the tensor for visualization.

        Args:
            tensor (torch.Tensor): The input tensor from the target layer.
            num_frames (int): Number of frames in the video.
            height (int): Height of the spatial patches.
            width (int): Width of the spatial patches.

        Returns:
            torch.Tensor: Reshaped tensor with dimensions (frames, height, width, channels).
        """
        # Ensure the tensor has the expected dimensions
        if tensor.dim() != 3:
            raise ValueError(f"Expected tensor with 3 dimensions, got {tensor.dim()} dimensions")

        # Reshape the tensor to separate frames and patches
        batch_size, num_tokens, channels = tensor.size()
        if num_tokens != num_frames * height * width:
            raise ValueError(f"Number of tokens ({num_tokens}) does not match expected dimensions ({num_frames} * {height} * {width})")

        # Reshape to (batch_size, num_frames, height, width, channels)
        result = tensor.view(batch_size, num_frames, height, width, channels)
        result = result.transpose(2, 3).transpose(1, 2)  # Rearrange dimensions to (batch_size, height, width, frames, channels)
        return result

    # Function to compute the Grad-CAM heatmap
    def __call__(self, inputs):
        self.model.zero_grad()  # Zero the gradients
        output = self.model(inputs)  # Forward pass

        # Get the index of the highest score in the output
        index = np.argmax(output.cpu().data.numpy())
        target = output[0][index]  # Get the target score
        target.backward()  # Backward pass to compute gradients

        # Get the gradients and features
        gradient = self.gradient[0].cpu().data.numpy()
        weight = np.mean(gradient, axis=(1, 2))  # Average the gradients
        feature = self.feature[0].cpu().data.numpy()

        # Compute the weighted sum of the features
        cam = feature * weight[:, np.newaxis, np.newaxis]
        cam = np.sum(cam, axis=0)  # Sum over the channels
        cam = np.maximum(cam, 0)  # Apply ReLU to remove negative values

        # Normalize the heatmap
        cam -= np.min(cam)
        cam /= np.max(cam)
        cam = cv2.resize(cam, (224, 224))  # Resize to match the input image size
        return cam  # Return the Grad-CAM heatmap
