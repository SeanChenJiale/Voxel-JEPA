# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import math
import pdb
from functools import partial
from typing import Union, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


from src.models.utils.pos_embs import get_3d_sincos_pos_embed 
from src.models.utils.modules import Block
from src.utils.tensors import repeat_interleave_batch
from src.masks.utils import apply_masks

from src.utils.tensors import trunc_normal_
# from timm.models.layers import trunc_normal_

class VideoDecoder(nn.Module):
    """Video Decoder that reconstructs video frames from V-JEPA features"""

    def __init__(
            self,
            embed_dim=768,
            decoder_dim=512,
            depth=6,
            num_heads=8,
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            norm_layer=nn.LayerNorm,
            init_std=0.02,
            patch_size=16,
            num_frames=16,
            tubelet_size=2,
            crop_size=224,
            in_chans=3,
            **kwargs
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.crop_size = crop_size
        self.in_chans = in_chans

        # Calculate spatial and temporal dimensions
        self.spatial_size = crop_size // patch_size
        self.temporal_size = num_frames // tubelet_size
        self.num_patches = self.spatial_size * self.spatial_size * self.temporal_size

        # Project V-JEPA features to decoder dimension
        self.feature_proj = nn.Linear(embed_dim, decoder_dim, bias=True)

        # Positional embedding for decoder
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, decoder_dim),
            requires_grad=False)

        # Transformer blocks for feature processing
        self.blocks = nn.ModuleList([
            Block(
                dim=decoder_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                act_layer=nn.GELU,
                grid_size=self.spatial_size,
                grid_depth=self.temporal_size,
                attn_drop=attn_drop_rate,
                norm_layer=norm_layer)
            for i in range(depth)])

        self.norm = norm_layer(decoder_dim)

        # Final projection to pixel space
        self.pixel_proj = nn.Linear(decoder_dim, patch_size * patch_size * tubelet_size * in_chans, bias=True)

        # Initialize weights
        self._init_pos_embed(self.pos_embed.data)
        self.init_std = init_std
        self.apply(self._init_weights)
        self._rescale_blocks()

    def _init_pos_embed(self, pos_embed):
        embed_dim = pos_embed.size(-1)
        sincos = get_3d_sincos_pos_embed(
            embed_dim,
            self.spatial_size,
            self.temporal_size,
            cls_token=False,
            uniform_power=False
        )
        pos_embed.copy_(torch.from_numpy(sincos).float().unsqueeze(0))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _rescale_blocks(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def forward(self, features):
        """
        Args:
            features: V-JEPA features of shape [B, N, embed_dim]
        Returns:
            reconstructed_video: Video frames of shape [B, C, T, H, W]
        """
        B, N, D = features.shape

        assert N == self.num_patches, f"Expected {self.num_patches} patches, got {N}"

        # Project features to decoder dimension
        x = self.feature_proj(features)  # [B, N, decoder_dim]

        # Add positional embedding
        x = x + self.pos_embed  # [B, N, decoder_dim]

        # Process through transformer blocks
        for blk in self.blocks:
            x = blk(x)  # [B, N, decoder_dim]

        x = self.norm(x)  # [B, N, decoder_dim]

        # Project to pixel space
        x = self.pixel_proj(x)  # [B, N, patch_pixels]

        # Reshape to video format
        patch_dim = self.patch_size * self.patch_size * self.tubelet_size * self.in_chans
        x = x.view(B, self.temporal_size, self.spatial_size, self.spatial_size, patch_dim)

        # Reshape patches to spatial dimensions
        x = x.view(B, self.temporal_size, self.spatial_size, self.spatial_size,
                   self.tubelet_size, self.patch_size, self.patch_size, self.in_chans)

        # Rearrange dimensions to [B, C, T, H, W]
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous()  # [B, C, T, H, W]
        x = x.view(B, self.in_chans, self.num_frames, self.crop_size, self.crop_size)

        return x


class TransformerDecoder(nn.Module):
    """
    MAE-style transformer decoder.
    Takes both unmasked (context) and masked tokens, processes them through transformer,
    and reconstructs pixel values.
    """
    def __init__(
        self,
        embed_dim: int = 768,
        decoder_dim: int = 512,
        patch_size: Union[Sequence[int], int] = 16,
        num_frames: int = 16,
        tubelet_size: int = 2,
        in_chans: int = 3,
        decoder_depth: int = 4,
        decoder_heads: int = 12,
        uniform_power: bool = False,
        **kwargs
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.patch_size = patch_size if isinstance(patch_size, (list, tuple)) else (patch_size, patch_size)
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.in_chans = in_chans
        
        # Calculate number of patches
        self.input_size = 224  # Assuming standard crop_size
        grid_size = self.input_size // self.patch_size[0]
        grid_depth = self.num_frames // self.tubelet_size
        self.num_patches = grid_depth * grid_size * grid_size
        
        # Number of pixels per patch
        self.pixel_values_per_patch = (
            self.patch_size[0] * self.patch_size[1] * self.tubelet_size * self.in_chans
        )
        
        # Project encoded features to decoder dimension
        self.enc_to_dec = nn.Linear(embed_dim, decoder_dim, bias=True)
        
        # Learnable mask tokens for masked positions
        self.mask_tokens = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        
        # Positional embeddings for decoder
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, decoder_dim),
            requires_grad=False
        )
        
        # Transformer blocks
        self.decoder_blocks = nn.ModuleList([
            Block(
                dim=decoder_dim,
                num_heads=decoder_heads,
                mlp_ratio=4.0,
                qkv_bias=True,
                norm_layer=nn.LayerNorm,
                grid_size=grid_size,
                grid_depth=grid_depth,
            )
            for i in range(decoder_depth)
        ])
        
        # Final layer norm and projection to pixels
        self.decoder_norm = nn.LayerNorm(decoder_dim)
        self.decoder_pred = nn.Linear(decoder_dim, self.pixel_values_per_patch, bias=True)
        
        # Initialize weights
        self._init_pos_embed()
        self.apply(self._init_weights)
    
    def _init_pos_embed(self):
        """Initialize positional embeddings with sincos"""
        pos_embed = self.decoder_pos_embed
        embed_dim = pos_embed.size(-1)
        grid_size = self.input_size // self.patch_size[0]
        grid_depth = self.num_frames // self.tubelet_size
        
        sincos = get_3d_sincos_pos_embed(
            embed_dim,
            grid_size,
            grid_depth,
            cls_token=False,
            uniform_power=False
        )
        pos_embed.copy_(torch.from_numpy(sincos).float().unsqueeze(0))
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, encoded_context, encoded_target,  masks_ctxt, masks_tgt):
        """
        Args:
            encoded_context: Context features from encoder [B, N_ctxt, embed_dim]
            encoded_target: Target features from predictor [B, N_tgt, embed_dim]
            masks_ctxt: Indices of context tokens [B, N_ctxt]
            masks_tgt: Indices of target tokens [B, N_tgt]
        
        Returns:
            pred_pixels: Predicted pixel values [B, N_tgt, pixel_dim]
        """
        B = encoded_context.shape[0]

        # Project to decoder dimension
        context_tokens = self.enc_to_dec(encoded_context)  # [B, N_ctxt, decoder_dim]
        _, N_ctxt, decoder_dim = context_tokens.shape
        
        # Add positional embeddings to context
        if self.decoder_pos_embed is not None:
            ctxt_pos_embed = self.decoder_pos_embed.repeat(B, 1, 1)
            if masks_ctxt is not None:
                ctxt_pos_embed = apply_masks(ctxt_pos_embed, [masks_ctxt], concat=False)[0]
            context_tokens += ctxt_pos_embed
        
        # Create mask tokens for target positions
        mask_tokens = self.mask_tokens.repeat(B, self.num_patches, 1)  # [B, total_N, decoder_dim]
        if masks_tgt is not None:
            batch_range = torch.arange(B, device=encoded_target.device)[:, None]
            target_mask_tokens = mask_tokens[batch_range, masks_tgt]  # [B, N_tgt, decoder_dim]
        else:
            target_mask_tokens = mask_tokens
        
        # Add positional embeddings to target mask tokens
        if self.decoder_pos_embed is not None and masks_tgt is not None:
            tgt_pos_embed = self.decoder_pos_embed.repeat(B, 1, 1)
            tgt_pos_embed = apply_masks(tgt_pos_embed, [masks_tgt], concat=False)[0]
            target_mask_tokens += tgt_pos_embed
        
        # Concatenate context and target tokens
        x = torch.cat([context_tokens, target_mask_tokens], dim=1)  # [B, N_ctxt+N_tgt, decoder_dim]
        
        # Combine masks for attention
        masks_combined = torch.cat([masks_ctxt, masks_tgt], dim=1) if masks_tgt is not None else None

        # Forward through transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x, mask=None) #masks_combined)
        x = self.decoder_norm(x)
        
        # Extract only target token predictions and project to pixels
        target_tokens = x[:, N_ctxt:]  # [B, N_tgt, decoder_dim]
        pred_pixels = self.decoder_pred(target_tokens)  # [B, N_tgt, pixel_dim]
        
        return pred_pixels


class LinearDecoder(nn.Module):
    """
    Minimal linear projection decoder.
    Maps masked encoded tokens directly to pixel patch values.
    Useful for lightweight reconstruction or ablations.
    """
    def __init__(
            self,
            embed_dim: int = 768,
            patch_size: Union[Sequence[int], int] = 16,
            num_frames: int = 16,
            tubelet_size: int = 2,
            in_chans: int = 3,
            **kwargs
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size if isinstance(patch_size, (list, tuple)) else (patch_size, patch_size)
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.in_chans = in_chans

        # Number of pixels per patch (tubelet)
        self.pixel_values_per_patch = (
                self.patch_size[0] * self.patch_size[1] * self.tubelet_size * self.in_chans
        )

        # Simple projection layer
        self.to_pixels = nn.Linear(embed_dim, self.pixel_values_per_patch)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, encoded_features, masked_indices=None):
        """
        Args:
            encoded_features: [B, N, D]
            masked_indices: [B, N_masked] or None
        Returns:
            pred_pixel_values: [B, N_masked, patch_dim]
        """
        batch_size = encoded_features.shape[0]

        if masked_indices is not None:
            batch_range = torch.arange(batch_size, device=encoded_features.device)[:, None]
            encoded_mask_tokens = encoded_features[batch_range, masked_indices]
        else:
            encoded_mask_tokens = encoded_features

        pred_pixel_values = self.to_pixels(encoded_mask_tokens)
        return pred_pixel_values




class VJEPADecoder(nn.Module):
    """
    Main decoder wrapper for V-JEPA
    Supports both linear and transformer-based MAE-style decoders
    """
    def __init__(
        self,
        encoder_embed_dim: int = 768,
        predictor_embed_dim: int = 384,
        patch_size: Union[Sequence[int], int] = 16,
        num_frames: int = 16,
        tubelet_size: int = 2,
        in_chans: int = 3,
        decoder_type: str = "linear",  # "transformer" or "linear"
        decoder_depth: int = 4,
        decoder_heads: int = 12,
        decoder_dim: int = 512,
        **kwargs
    ):
        super().__init__()
        
        self.decoder_type = decoder_type

        if decoder_type == "linear":
            self.decoder = LinearDecoder(
                embed_dim=encoder_embed_dim,
                patch_size=patch_size,
                num_frames=num_frames,
                tubelet_size=tubelet_size,
                in_chans=in_chans,
                **kwargs
            )
        elif decoder_type == "transformer":
            self.decoder = TransformerDecoder(
                embed_dim=encoder_embed_dim,
                decoder_dim=decoder_dim,
                patch_size=patch_size,
                num_frames=num_frames,
                tubelet_size=tubelet_size,
                in_chans=in_chans,
                decoder_depth=decoder_depth,
                decoder_heads=decoder_heads,
                **kwargs
            )
        elif decoder_type == "videodecoder":
            self.decoder = VideoDecoder(
                embed_dim=encoder_embed_dim,
                decoder_dim=decoder_dim,
                depth=decoder_depth,
                num_heads=decoder_heads,
                patch_size=patch_size,
                num_frames=num_frames,
                tubelet_size=tubelet_size,
                crop_size=224,
                in_chans=in_chans,
                **kwargs
            )
        else:
            raise ValueError(
                f"Unknown decoder type: {self.decoder_type}. Must be one of "
                f"'linear', 'transformer', or 'videodecoder'."
            )
    
    def forward(self, encoded_features, masked_indices=None, encoded_context=None, masks_ctxt=None, masks_tgt=None):
        """
        Args:
            encoded_features: For linear decoder - encoded features from predictor [B, N, embed_dim]
            masked_indices: For linear decoder - indices of masked tokens [B, num_masked]
            
            encoded_context: For transformer decoder - context features from encoder [B, N_ctxt, embed_dim]
            encoded_target: For transformer decoder - target features from predictor [B, N_tgt, embed_dim]
            masks_ctxt: For transformer decoder - indices of context tokens [B, N_ctxt]
            masks_tgt: For transformer decoder - indices of target tokens [B, N_tgt]
        
        Returns:
            pred_pixel_values: Predicted pixel values [B, num_masked, patch_dim]
        """
        if self.decoder_type == "linear":
            return self.decoder(encoded_features, masked_indices)
        elif self.decoder_type == "transformer":
            # Transformer decoder needs both context and target
            if encoded_context is None or encoded_features is None:
                raise ValueError("Transformer decoder requires both encoded_context and encoded_target")
            return self.decoder(encoded_context, encoded_features, masks_ctxt, masks_tgt)
        elif self.decoder_type == "videodecoder":
            # VideoDecoder expects a single full token grid [B, N, D]
            # (e.g., your merged_features)
            return self.decoder(encoded_features)
        else:
            raise ValueError(f"Unknown decoder type: {self.decoder_type}")



if __name__ == "__main__":
    # Test the decoder
    batch_size = 2
    num_patches = 196  # 14x14 patches
    embed_dim = 768
    num_masked = 50
    
    # Create test data
    encoded_features = torch.randn(batch_size, num_patches, embed_dim)
    masked_indices = torch.randint(0, num_patches, (batch_size, num_masked))
    
    # Test pixel decoder

