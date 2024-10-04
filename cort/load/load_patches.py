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


def load_corner_patches(
    patches_to_load: list[str],
    report_progress=False,
    max_width: int = 128,
    max_height: int = 256,
) -> list[cort.CorticalPatch]:
    """Load all patches from a base path."""
    patches = []
    if report_progress:
        patches_to_load = tqdm(patches_to_load, desc="Loading corner patches")

    for patch_path in patches_to_load:
        patch = cort.load.load_corner_patch(
            patch_path, max_width=max_width, max_height=max_height
        )
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


def load_mask_patches(
    base_path: str,
    patches_to_load: list[str] = None,
    report_progress=False,
    max_width: int = 128,
    max_height: int = 256,
) -> list[cort.MaskCorticalPatch]:
    """Load all patches from a base path."""
    patches = []

    if patches_to_load is None:
        patches_to_load = cort.load.find_all_corner_patches(base_path)

    if report_progress:
        patches_to_load = tqdm(patches_to_load, desc="Loading mask patches")

    for patch_path in patches_to_load:
        patch = cort.load.load_mask_patch(
            patch_path, max_width=max_width, max_height=max_height
        )
        patches.append(patch)

    return patches


def load_pre_mask_patches(
    base_path: str,
    patches_to_load: list[str] = None,
    report_progress=False,
) -> list[cort.MaskCorticalPatch]:
    """Load all patches from a base path."""
    patches = []

    if patches_to_load is None:
        patches_to_load = cort.load.find_all_preprocessed(base_path)

    if report_progress:
        patches_to_load = tqdm(patches_to_load, desc="Loading mask patches")

    for patch_path in patches_to_load:
        patch = cort.MaskCorticalPatch.load(base_path, patch_path)
        patches.append(patch)

    return patches
