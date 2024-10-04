import numpy as np
import torch

import volume

LIGHT_DIR = torch.tensor([0, 1, 1], device="cuda", dtype=torch.float)
CUTOUT_DIR = torch.tensor([-1, 0, 0], device="cuda", dtype=torch.float)

def render_tensor(
        prerended: volume.PrerenderedVolume,
        angles: tuple[float, float, float],
        cutout_perc: tuple[float, float, float]
) -> np.ndarray:
    rot_mat = volume.create_rotation_matrix_tensor(
        torch.tensor([angles[0]], device="cuda", dtype=torch.float),
        torch.tensor([angles[1]], device="cuda", dtype=torch.float),
        torch.tensor([angles[2]], device="cuda", dtype=torch.float)
    )

    rev_rot_mat =volume.create_rotation_matrix_tensor(
        torch.tensor([angles[2]], device="cuda", dtype=torch.float),
        torch.tensor([angles[1]], device="cuda", dtype=torch.float),
        torch.tensor([angles[0]], device="cuda", dtype=torch.float)
    )

    prerended = volume.light_and_cutout(prerended, cutout_perc, rev_rot_mat, LIGHT_DIR, CUTOUT_DIR)

    rotated = volume.affine(prerended.volume, rot_mat, mode="nearest", padding_mode="zeros")

    alpha = rotated[-1, None, :, :, :]
    transparency = 1 - rotated[-1, None, :, :, :]

    values = rotated[:-1, :, :, :]

    cum_prod_transparency = torch.cumprod(transparency, dim=1).roll(1, dims=1)
    cum_prod_transparency[:, 0, :, :] = 1

    cum_prod_values = values * alpha * cum_prod_transparency

    output = torch.sum(cum_prod_values, dim=1)

    img = torch.clip(output, 0, 1).permute(1, 2, 0).cpu().numpy()

    img = (img * 255).astype(np.uint8)

    return img
