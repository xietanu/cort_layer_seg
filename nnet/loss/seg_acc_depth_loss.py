import datasets
import nnet.loss

import torch


def seg_acc_depth_loss(
    outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ground_truths: datasets.datatypes.SegGroundTruths,
    acc_gts: torch.Tensor,
    ignore_index: int,
) -> torch.Tensor:
    """Compute the combined segmentation and depth loss."""
    seg_preds, acc_preds, depth_preds = outputs

    seg_loss = nnet.loss.tversky_loss(
        seg_preds, ground_truths.segmentation, ignore_index
    )
    acc_loss = torch.nn.functional.mse_loss(acc_preds.squeeze(), acc_gts.squeeze())
    depth_loss = nnet.loss.depthmap_loss(
        depth_preds, ground_truths.depth_maps, ground_truths.segmentation, ignore_index
    )

    return seg_loss + depth_loss + acc_loss
