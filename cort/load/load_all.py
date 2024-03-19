import os

from tqdm import tqdm

import cort
import cort.load


def load_all_patches(
    base_path: str, downscale: float = 1, report_progress=False
) -> list[cort.CorticalPatch]:
    """Load all patches from a base path."""
    patches_to_load = find_all_patches(base_path)

    patches = []
    if report_progress:
        patches_to_load = tqdm(patches_to_load, desc="Loading patches")

    for patch_path in patches_to_load:
        patch = cort.load.load_patch(patch_path, downscale=downscale)
        patches.append(patch)

    return patches


def find_all_patches(base_path: str) -> list[str]:
    """Find all patches in a base path."""
    patches_to_load = []
    info_file = os.path.join(base_path, "info.txt")
    if os.path.isfile(info_file):
        patches_to_load = [base_path]
    else:
        for folder in os.listdir(base_path):
            folder_path = os.path.join(base_path, folder)
            if os.path.isdir(folder_path):
                patches_to_load.extend(find_all_patches(folder_path))
    return patches_to_load
