import os

import cort
import cort.load


def load_all_patches(base_path: str, downscale: float = 1) -> list[cort.CorticalPatch]:
    """Load all patches from a base path."""
    patches = []
    info_file = os.path.join(base_path, "info.txt")
    if os.path.isfile(info_file):
        patches = [cort.load.load_patch(base_path, downscale=downscale)]
    else:
        for folder in os.listdir(base_path):
            folder_path = os.path.join(base_path, folder)
            if os.path.isdir(folder_path):
                patches.extend(load_all_patches(folder_path, downscale=downscale))
                
    return patches
