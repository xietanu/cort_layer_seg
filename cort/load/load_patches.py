from tqdm import tqdm

import cort
import cort.load


def load_patches(
    patches_to_load: list[str], downscale: float = 1, report_progress=False
) -> list[cort.CorticalPatch]:
    """Load all patches from a base path."""
    patches = []
    if report_progress:
        patches_to_load = tqdm(patches_to_load, desc="Loading patches")

    for patch_path in patches_to_load:
        patch = cort.load.load_patch(patch_path, downscale=downscale)
        patches.append(patch)

    return patches


def load_preprocessed_patches(
    base_path: str, patches_to_load: list[str] = None, report_progress=False
) -> list[cort.CorticalPatch]:
    """Load all preprocessed patches from a base path."""
    patches = []

    if patches_to_load is None:
        patches_to_load = cort.load.find_all_preprocessed(base_path)

    if report_progress:
        patches_to_load = tqdm(patches_to_load, desc="Loading patches")

    for patch_path in patches_to_load:
        patch = cort.CorticalPatch.load(base_path, patch_path)
        patches.append(patch)

    return patches
