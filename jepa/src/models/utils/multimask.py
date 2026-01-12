# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch.nn as nn


class MultiMaskWrapper(nn.Module):

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x, masks=None):
        if masks is None:
            return self.backbone(x)

        if (masks is not None) and not isinstance(masks, list):
            masks = [masks]
        outs = []
        for m in masks:
            outs += [self.backbone(x, masks=m)]
        return outs


class PredictorMultiMaskWrapper(nn.Module):

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, ctxt, tgt, masks_ctxt, masks_tgt):
        if type(ctxt) is not list:
            ctxt = [ctxt]
        if type(tgt) is not list:
            tgt = [tgt]
        if type(masks_ctxt) is not list:
            masks_ctxt = [masks_ctxt]
        if type(masks_tgt) is not list:
            masks_tgt = [masks_tgt]

        outs = []
        for i, (zi, hi, mc, mt) in enumerate(zip(ctxt, tgt, masks_ctxt, masks_tgt)):
            outs += [self.backbone(zi, hi, mc, mt, mask_index=i)]
        return outs


class DecoderMultiMaskWrapper(nn.Module):
    """
    Wrapper for decoder to handle multiple mask blocks (like PredictorMultiMaskWrapper)
    """

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, encoded_features, encoded_context=None, masks_ctxt=None, masks_tgt=None, masked_indices=None):
        """
        Handles both linear and transformer decoders with multiple mask blocks
        
        Args:
            encoded_features: Target features from predictor (for both decoder types)
            encoded_context: Context features from encoder (for transformer decoder)
            masks_ctxt: Indices of context tokens
            masks_tgt: Indices of target tokens (same as multiple mask blocks)
            masked_indices: For linear decoder compatibility
        """
        # Handle list of masks (multiple mask blocks)
        if masks_tgt is not None and isinstance(masks_tgt, list):
            # Multiple mask blocks
            if isinstance(encoded_features, list):
                # encoded_features is list of [B, N_tgt_i, D] for each mask
                targets = encoded_features
            else:
                # Single target to be reused for all masks
                targets = [encoded_features] * len(masks_tgt)
            
            if isinstance(encoded_context, list):
                contexts = encoded_context
            else:
                contexts = [encoded_context] * len(masks_tgt) if encoded_context is not None else [None] * len(masks_tgt)
            
            if masks_ctxt is not None and isinstance(masks_ctxt, list):
                ctx_masks = masks_ctxt
            else:
                ctx_masks = [masks_ctxt] * len(masks_tgt)
            
            outs = []
            for target, context, mc, mt in zip(targets, contexts, ctx_masks, masks_tgt):
                out = self._forward_single_block(target, context, mc, mt, masked_indices)
                outs.append(out)
            return outs
        else:
            # Single mask block
            return self._forward_single_block(encoded_features, encoded_context, masks_ctxt, masks_tgt, masked_indices)
    
    def _forward_single_block(self, encoded_features, encoded_context, masks_ctxt, masks_tgt, masked_indices):
        """Forward pass for a single mask block"""
        if self.backbone.decoder_type == "transformer":
            return self.backbone(
                encoded_features=encoded_features,
                encoded_context=encoded_context,
                masks_ctxt=masks_ctxt,
                masks_tgt=masks_tgt
            )
        else:
            return self.backbone(encoded_features, masked_indices)
