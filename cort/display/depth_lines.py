import numpy as np
import torch


DEPTH_CUTOFF = 0.75


def draw_max_depth(
    depth_maps: torch.Tensor | np.ndarray, bg_img: torch.Tensor | np.ndarray = None
):
    if isinstance(depth_maps, torch.Tensor):
        depth_maps = depth_maps.detach().cpu().numpy()

    depth_maps = depth_maps.squeeze()

    depth_lines = np.ones((depth_maps.shape[1], depth_maps.shape[2], 3))

    if bg_img is not None:
        if isinstance(bg_img, torch.Tensor):
            bg_img = bg_img.detach().cpu().numpy()

        bg_img = bg_img.squeeze()
        depth_lines[:, :, 0] = bg_img
        depth_lines[:, :, 1] = bg_img
        depth_lines[:, :, 2] = bg_img

    max_depth = (1 - np.abs(depth_maps).min(axis=0)) ** 2

    depth_lines[:, :, 0] = np.maximum(max_depth, depth_lines[:, :, 0])
    depth_lines[:, :, 2] *= 1 - max_depth
    depth_lines[:, :, 1] = depth_lines[:, :, 2]

    depth_lines = np.clip(depth_lines, 0, 1)

    return depth_lines


def draw_depth_lines(
    depth_maps: torch.Tensor | np.ndarray, bg_img: torch.Tensor | np.ndarray = None
) -> np.ndarray:
    """Draw depth lines on the depth map."""
    if isinstance(depth_maps, torch.Tensor):
        depth_maps = depth_maps.detach().cpu().numpy()

    depth_maps = depth_maps.squeeze()

    depth_lines = np.zeros((depth_maps.shape[1], depth_maps.shape[2], 3))

    if bg_img is not None:
        if isinstance(bg_img, torch.Tensor):
            bg_img = bg_img.detach().cpu().numpy()

        bg_img = bg_img.squeeze()
        depth_lines[:, :, 0] = bg_img
        depth_lines[:, :, 1] = bg_img
        depth_lines[:, :, 2] = bg_img

    for i in range(depth_maps.shape[0]):
        depth_lines[:, :, 0] += (
            (depth_maps[i] > -DEPTH_CUTOFF) * (depth_maps[i] < DEPTH_CUTOFF)
        ).astype(np.float32)
        depth_lines[:, :, 1] -= (
            (depth_maps[i] > DEPTH_CUTOFF) * (depth_maps[i] < DEPTH_CUTOFF)
        ).astype(np.float32)
        depth_lines[:, :, 2] -= (
            (depth_maps[i] > DEPTH_CUTOFF) * (depth_maps[i] < DEPTH_CUTOFF)
        ).astype(np.float32)
    depth_lines = np.clip(depth_lines, 0, 1)
    return depth_lines


def draw_depth_seg(depth_maps: torch.Tensor | np.ndarray):
    if isinstance(depth_maps, torch.Tensor):
        depth_maps = depth_maps.detach().cpu().numpy()

    depth_maps = depth_maps.squeeze()

    depth_seg = np.zeros((depth_maps.shape[1], depth_maps.shape[2]), dtype=np.uint8)

    for i in range(depth_maps.shape[0]):
        depth_seg[depth_maps[i] > 0] = i + 1

    return depth_seg
