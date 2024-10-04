import numpy as np
import torch

import volume


def light_and_cutout(vol: volume.PrerenderedVolume, cutout_perc: tuple[float, float, float], rot_mat: torch.Tensor, light_dir: torch.Tensor, cutout_dir: torch.Tensor):
    cur_volume = vol.volume.clone()

    mags = [
        torch.dot(light_dir.float(), direction.float() @ rot_mat[:3, :3].T)
        for direction in vol.light_dirs
    ]

    for mask, mag in zip(vol.light_masks, mags):
        cur_volume[:3, mask] *= 1 + 0.3 * mag

    cutout_mags = torch.stack([
        torch.dot(cutout_dir.float(), corner.float() @ rot_mat[:3, :3].T)
        for corner in vol.corners
    ])

    best_corner = vol.corners[torch.argmax(cutout_mags)]


    cutout_mask = torch.zeros_like(vol.alpha_mask, device="cuda", dtype=torch.uint8)

    height_pos = int(cutout_mask.shape[0] * cutout_perc[0])
    width_pos = int(cutout_mask.shape[1] * cutout_perc[1])
    depth_pos = int(cutout_mask.shape[2] * cutout_perc[2])

    height_pos = np.clip(height_pos, 1, cutout_mask.shape[0] - 1)
    width_pos = np.clip(width_pos, 1, cutout_mask.shape[1] - 1)
    depth_pos = np.clip(depth_pos, 1, cutout_mask.shape[2] - 1)

    if best_corner[0] == 1:
        cutout_mask[height_pos, :, :] = 3
    else:
        cutout_mask[height_pos - 1, :, :] = 2
    if best_corner[1] == 1:
        cutout_mask[:, width_pos, :] = 5
    else:
        cutout_mask[:, width_pos - 1, :] = 4
    if best_corner[2] == 1:
        cutout_mask[:, :, depth_pos] = 7
    else:
        cutout_mask[:, :, depth_pos - 1] = 6

    if best_corner[0] == 1:
        cutout_mask[:height_pos, :, :] = 1
    else:
        cutout_mask[height_pos:, :, :] = 1
    if best_corner[1] == 1:
        cutout_mask[:, :width_pos, :] = 1
    else:
        cutout_mask[:, width_pos:, :] = 1
    if best_corner[2] == 1:
        cutout_mask[:, :, :depth_pos] = 1
    else:
        cutout_mask[:, :, depth_pos:] = 1

    cur_volume[-1, cutout_mask == 0] = 0
    for i, mag in enumerate(mags, 2):
        cur_volume[:3, cutout_mask == i] *= 1 + 0.3 * mag

    #short_rot_mat = rot_mat[:3, :3]

    #inv_rot_mat = torch.linalg.inv(short_rot_mat)


    #for i in np.linspace(0, 0.5, 25):
    #    dot = (-light_dir @ inv_rot_mat.T)

    #    dot = (torch.tensor([cur_volume.shape[1]*0.5, cur_volume.shape[2]*0.5, cur_volume.shape[3]*0.5], dtype=torch.float, device="cuda") + torch.tensor([cur_volume.shape[1]*i, cur_volume.shape[1]*i, cur_volume.shape[1]*i], dtype=torch.float, device="cuda") * dot).int()

    #print(dot)

    #    if (0 <= dot[0].item() < cur_volume.shape[1]) and (0 <= dot[1].item() < cur_volume.shape[2]) and (
    #            0 <= dot[2].item() < cur_volume.shape[3]):
    #        cur_volume[:, dot[0], dot[1], dot[2]] = torch.tensor([1, 0, 0, 1], dtype=torch.float, device="cuda")

    return volume.PrerenderedVolume(cur_volume, vol.alpha_mask, vol.light_masks, vol.light_dirs, vol.corners)
