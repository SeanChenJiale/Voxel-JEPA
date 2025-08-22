# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import pdb

import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from functools import reduce
from operator import mul
import torch


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding
    """
    def __init__(
        self,
        patch_size=16,
        in_chans=3,
        embed_dim=768
    ):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class PatchEmbed3D(nn.Module):
    """
    Image to Patch Embedding
    """

    def __init__(
        self,
        patch_size=16,
        tubelet_size=2,
        in_chans=3,
        embed_dim=768,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size

        self.proj = nn.Conv3d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
        )

    def forward(self, x, **kwargs):
        B, C, T, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class BlockwisePatchEmbed3D(nn.Module):
    """
    Blockwise 3D Patch Embedding: split video into temporal blocks
    and embed each separately.
    """

    def __init__(
        self,
        patch_size=16,
        tubelet_size=2,
        num_frames=16,
        in_chans=3,
        embed_dim=768,
    ):
        super().__init__()
        assert (
            num_frames % tubelet_size == 0
        ), f"num_frames={num_frames} not divisible by tubelet_size={tubelet_size}"

        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.embed_dim = embed_dim
        self.num_blocks = num_frames // tubelet_size

        # Each patch's raw dimension: tubelet_size × patch_H × patch_W × channels
        self.patch_dim = in_chans * tubelet_size * patch_size * patch_size

        self.pre_norm = nn.LayerNorm(self.patch_dim)
        self.post_norm = nn.LayerNorm(embed_dim)

        """
        This moves t1 (tubelet_size) into the patch dimension group (c t1 p1 p2), making patch_dim = 3 * 2 * 16 * 16 = 1536.
        (h w) = 14 * 14 = 196
        Output shape after to_patch: (36, 8, 196, 1536)
        """
        self.to_patch = Rearrange(
            "b c (tb t1) (h p1) (w p2) -> b tb (h w) (c t1 p1 p2)",
            tb=self.num_blocks,
            p1=self.patch_size,
            p2=self.patch_size,
            t1=self.tubelet_size,
        )

        # self.to_patch = nn.Sequential(
        #     Rearrange(
        #         "b c (tb t1) (h p1) (w p2) -> b tb t1 h w (c p1 p2)",
        #         tb=self.num_blocks, p1=self.patch_size, p2=self.patch_size, t1=self.tubelet_size
        #     ),
        #     Rearrange(
        #         "b tb t1 h w x -> b tb (t1 h w) x"
        #     )
        # )

        self.blockwise_embed = nn.ModuleList(
            [nn.Linear(self.patch_dim, embed_dim) for _ in range(self.num_blocks)]
        )

    def embed(self, patches):
        patches = self.pre_norm(patches)
        embeds = [self.blockwise_embed[i](patches[:, i, :, :]) for i in range(self.num_blocks)]
        embeds = torch.stack(embeds, dim=1)
        embeds = rearrange(embeds, "b g n d -> b (g n) d")
        embeds = self.post_norm(embeds)
        return embeds


    def forward(self, x):
        """
        Args:
            x: (B, C, T, H, W) => torch.Size([36, 3, 16, 224, 224])
        Returns:
            (B, num_patches_total, embed_dim)
        """
        patches = self.to_patch(x)  # patches.shape: torch.Size([36, 8, 196, 1536])
        return self.embed(patches)

