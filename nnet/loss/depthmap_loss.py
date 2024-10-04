import torch


def depthmap_loss(
    depth_preds: torch.Tensor,
    depth_targets: torch.Tensor,
    seg_targets: torch.Tensor,
    ignore_index: int,
) -> torch.Tensor:
    not_ignore_mask = (seg_targets != ignore_index).expand_as(depth_preds)

    n_layers = depth_preds.size(1)

    depth_preds_not_ignored = depth_preds[not_ignore_mask].float()
    depth_targets_not_ignored = depth_targets[not_ignore_mask].float()

    depth_loss = torch.nn.functional.mse_loss(
        depth_preds_not_ignored, depth_targets_not_ignored
    )

    return depth_loss / n_layers
