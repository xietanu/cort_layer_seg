import numpy as np

import cort


def pad_patch(
    patch: cort.CorticalPatch, desired_size: tuple[int, int]
) -> cort.CorticalPatch:
    """Pad a patch to a desired size."""
    if patch.image.shape[0] > desired_size[0] or patch.image.shape[1] > desired_size[1]:
        raise ValueError(
            f"Patch is larger than desired size: {patch.image.shape} > {desired_size}"
        )

    new_image = np.zeros(desired_size)
    new_mask = np.ones(desired_size) * cort.constants.PADDING_MASK_VALUE
    new_image[: patch.image.shape[0], : patch.image.shape[1]] = patch.image
    new_mask[: patch.mask.shape[0], : patch.mask.shape[1]] = patch.mask
    return cort.CorticalPatch(
        image=new_image,
        mask=new_mask,
        inplane_resolution_micron=patch.inplane_resolution_micron,
        section_thickness_micron=patch.section_thickness_micron,
        brain_id=patch.brain_id,
        section_id=patch.section_id,
        patch_id=patch.patch_id,
        brain_area=patch.brain_area,
    )
