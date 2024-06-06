import itertools


import numpy as np


def create_border_flow_map(mask: np.ndarray, depth_maps: np.ndarray):
    """Create a map of the distance to the next border."""
    border_flow_map = np.min(depth_maps, axis=2).astype(np.float32) / 10
    # np.zeros_like(mask, dtype=np.float32)

    # for i in range(np.max(mask) + 1):
    #    mask_i = mask == i
    #    border_flow_map[mask_i] = i + depth_maps[mask_i, i] / (
    #        depth_maps[mask_i, i] + depth_maps[mask_i, i + 1] + 1e-6
    #    )

    return border_flow_map
