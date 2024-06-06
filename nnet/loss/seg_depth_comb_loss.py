import nnet.loss

import torch


def seg_depth_comb_loss(
    outputs: tuple[torch.Tensor, torch.Tensor],
    targets: tuple[torch.Tensor, torch.Tensor],
    ignore_index: int,
) -> torch.Tensor:
    """Compute the combined segmentation and depth loss."""
    seg_preds, depth_preds = outputs
    seg_targets, depth_targets = targets
    seg_loss = nnet.loss.tversky_loss(seg_preds, seg_targets, ignore_index)
    depth_loss = nnet.loss.depthmap_loss(
        depth_preds, depth_targets, seg_targets, ignore_index
    )

    return seg_loss + depth_loss
