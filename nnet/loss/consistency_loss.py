import torch
import nnet.loss


def consistency_loss_alt(
    seg_logits: torch.Tensor,
    depth_maps: torch.Tensor,
    gt_seg: torch.Tensor,
    ignore_index: int,
):
    depth_seg = depth_maps.ceil().sum(dim=1).long()

    mask = gt_seg == ignore_index

    depth_seg[mask.squeeze()] = ignore_index

    loss = torch.nn.functional.cross_entropy(
        seg_logits, depth_seg, reduction="mean", ignore_index=ignore_index
    )

    return loss


def consistency_loss(
    seg_logits: torch.Tensor,
    depth_maps: torch.Tensor,
    gt_seg: torch.Tensor,
    ignore_index: int,
):
    first_layer = torch.ones_like(depth_maps[:, 0]).unsqueeze(1)
    last_layer = -torch.ones_like(depth_maps[:, -1]).unsqueeze(1)
    combined_layers = torch.cat([first_layer, depth_maps, last_layer], dim=1)
    unnormalized_probs = (combined_layers[:, :-1] - combined_layers[:, 1:]) / 2

    unnormalized_probs = torch.clamp(unnormalized_probs, 0, 1)

    probs = unnormalized_probs / unnormalized_probs.sum(dim=1, keepdim=True)

    # print("probs:", probs.max(), probs.min())

    return nnet.loss.tversky_loss(
        seg_logits,
        probs,
        ignore_index=ignore_index,
        ignore_mask=gt_seg == ignore_index,
    )
