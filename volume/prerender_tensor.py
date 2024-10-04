import itertools

import numpy as np
import torch
import volume



def prerender_tensor(col_vol: np.ndarray):
    col_vol = torch.tensor(col_vol).permute(3, 0, 1, 2).cuda()

    alpha_mask = col_vol[3, :, :, :] > 0.1


    up_dir = torch.tensor([1, 0, 0], device="cuda", dtype=torch.float)
    down_dir = torch.tensor([-1, 0, 0], device="cuda", dtype=torch.float)
    left_dir = torch.tensor([0, 1, 0], device="cuda", dtype=torch.float)
    right_dir = torch.tensor([0, -1, 0], device="cuda", dtype=torch.float)
    forward_dir = torch.tensor([0, 0, 1], device="cuda", dtype=torch.float)
    backward_dir = torch.tensor([0, 0, -1], device="cuda", dtype=torch.float)

    corners = torch.tensor([
        [x, y, z]
        for x, y, z in itertools.product([-1, 1], repeat=3)
    ], device="cuda", dtype=torch.float)

    up_mask = torch.logical_xor(alpha_mask, torch.roll(alpha_mask, 1, dims=0))
    up_mask[0, :, :] = True
    up_mask[-1, :, :] = False

    down_mask = torch.logical_xor(alpha_mask, torch.roll(alpha_mask, -1, dims=0))
    down_mask[0, :, :] = False
    down_mask[-1, :, :] = True

    left_mask = torch.logical_xor(alpha_mask, torch.roll(alpha_mask, 1, dims=1))
    left_mask[:, 0, :] = True
    left_mask[:, -1, :] = False

    right_mask = torch.logical_xor(alpha_mask, torch.roll(alpha_mask, -1, dims=1))
    right_mask[:, 0, :] = False
    right_mask[:, -1, :] = True

    forward_mask = torch.logical_xor(alpha_mask, torch.roll(alpha_mask, 1, dims=2))
    forward_mask[:, :, 0] = True
    forward_mask[:, :, -1] = False

    backward_mask = torch.logical_xor(alpha_mask, torch.roll(alpha_mask, -1, dims=2))
    backward_mask[:, :, 0] = False
    backward_mask[:, :, -1] = True

    light_masks = [
        up_mask, down_mask, left_mask, right_mask, forward_mask, backward_mask
    ]

    light_dirs = [
        up_dir, down_dir, left_dir, right_dir, forward_dir, backward_dir
    ]

    return volume.PrerenderedVolume(col_vol, alpha_mask, light_masks, light_dirs, corners)
