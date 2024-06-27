import datasets
import nnet.loss

import torch


def seg_acc_loss(
    outputs: tuple[torch.Tensor, torch.Tensor],
    ground_truths: datasets.datatypes.SegGroundTruths,
    acc_gts: torch.Tensor,
    ignore_index: int,
) -> torch.Tensor:
    """Compute the combined segmentation and depth loss."""
    seg_preds, acc_preds = outputs

    seg_loss = nnet.loss.tversky_loss(
        seg_preds, ground_truths.segmentation, ignore_index
    )
    acc_loss = torch.nn.functional.mse_loss(acc_preds.squeeze(), acc_gts.squeeze())

    return seg_loss + acc_loss
