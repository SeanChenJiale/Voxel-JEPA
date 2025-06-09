import yaml
import os
import torch
import src.models.vision_transformer as vit
from src.models.attentive_pooler import AttentiveRegressor
from src.datasets.video_dataset import VideoDataset

# Path to the video file
video_path = "/local_data/sean_hasitha/sean/Dataset_AD_bin_classification_96/1/val/ADNI_MP_RAGE/2007-05-31_13_21_34.0_ADNI_14M4_TS_2_20070531132048_2/Axial.mp4"

# Initialize the VideoDataset
dataset = VideoDataset(
    data_paths=[video_path],  # Provide the video path as a list
    frames_per_clip=16,       # Number of frames per clip
    frame_step=4,             # Step size between frames
    num_clips=1,              # Number of clips to extract from the video
    random_clip_sampling=False,  # Disable random sampling for deterministic results
    allow_clip_overlap=False, # Disable clip overlap
    filter_short_videos=False # Do not filter short videos
)

# Load the first video clip (index 0)
buffer, label, clip_indices = dataset[0]

# Print the results
print("Buffer shape:", buffer[0].shape)  # Shape of the video clip (frames, height, width, channels)
print("Label:", label)                  # Label associated with the video
print("Clip indices:", clip_indices)    # Indices of the frames in the clip
# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the encoder (pretrained Vision Transformer)
encoder = vit.vit_large(
    img_size=224,  # Resolution used during training
    patch_size=16,
    num_frames=16,  # Frames per clip
    tubelet_size=2,
    uniform_power=False,
    use_sdpa=False,
    use_SiLU=False,
    tight_SiLU=True,
).to(device)

def load_pretrained(
    encoder,
    pretrained,
    checkpoint_key='target_encoder'
):
    print(f'Loading pretrained model from {pretrained}')
    checkpoint = torch.load(pretrained, map_location='cpu')
    try:
        pretrained_dict = checkpoint[checkpoint_key]
    except Exception:
        pretrained_dict = checkpoint['encoder']

    pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items()}
    pretrained_dict = {k.replace('backbone.', ''): v for k, v in pretrained_dict.items()}
    for k, v in encoder.state_dict().items():
        if k not in pretrained_dict:
            print(f'key "{k}" could not be found in loaded state dict')
        elif pretrained_dict[k].shape != v.shape:
            print(f'key "{k}" is of different shape in model and loaded state dict')
            pretrained_dict[k] = v
    msg = encoder.load_state_dict(pretrained_dict, strict=False)
    print(encoder)
    print(f'loaded pretrained model with msg: {msg}')
    print(f'loaded pretrained encoder from epoch: {checkpoint["epoch"]}\n path: {pretrained}')
    del checkpoint
    return encoder

# Load pretrained encoder weights (if needed)
encoder_checkpoint = '/local_data/sean_hasitha/sean/V_jepa_logs/32batchsize_vitl_96skip_scratch/vitl16.pth.tar'
encoder = load_pretrained(encoder, encoder_checkpoint)
encoder.eval()

# Initialize the classifier
regressor = AttentiveRegressor(
    embed_dim=encoder.embed_dim,
    num_heads=encoder.num_heads,
    depth=1,
      # Replace with the number of classes used during training
).to(device)

# Load the classifier weights
checkpoint = torch.load('/local_data/sean_hasitha/sean/V_jepa_logs/32batchsize_vitl_96skip_scratch/video_regression_frozen/vitl16_brainslice96_regressor/vitl16_96_regressor-latest.pth.tar', map_location='cpu')
# Remove "module." prefix from keys if present
state_dict = checkpoint['classifier']
state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

# Load the state dictionary into the model
missing_keys, unexpected_keys = regressor.load_state_dict(state_dict, strict=False)
print("Missing keys:", missing_keys)
print("Unexpected keys:", unexpected_keys)
regressor.eval()

# Example input: raw video data
input_data = torch.randn(1, 16, 3, 224, 224).to(device)  # (batch_size, frames, channels, height, width)

# Permute the input to match the expected shape: [batch_size, channels, frames, height, width]
input_data = input_data.permute(0, 2, 1, 3, 4)

# Pass the input through the encoder to get embeddings
with torch.no_grad():
    embeddings = encoder(input_data)  # Shape: [batch_size, seq_len, embed_dim]

# Pass the embeddings through the classifier
output = regressor(embeddings)
print(output)