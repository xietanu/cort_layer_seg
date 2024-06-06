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

    depth_maps_shape = (*desired_size, patch.depth_maps.shape[2])

    new_image = np.zeros(desired_size)
    new_image[: patch.image.shape[0], : patch.image.shape[1]] = patch.image

    new_mask = np.ones(desired_size) * cort.constants.PADDING_MASK_VALUE
    new_mask[: patch.mask.shape[0], : patch.mask.shape[1]] = patch.mask

    new_borders = np.zeros((desired_size[0], desired_size[1], patch.borders.shape[2]))
    new_borders[: patch.borders.shape[0], : patch.borders.shape[1], :] = patch.borders

    new_depth_maps = np.zeros(
        (desired_size[0], desired_size[1], patch.depth_maps.shape[2])
    )
    new_depth_maps[: patch.depth_maps.shape[0], : patch.depth_maps.shape[1], :] = (
        patch.depth_maps
    )

    new_siibra_image = np.zeros(desired_size)
    new_siibra_image[: patch.siibra_image.shape[0], : patch.siibra_image.shape[1]] = (
        patch.siibra_image
    )

    new_siibra_mask = np.ones(desired_size) * cort.constants.PADDING_MASK_VALUE
    new_siibra_mask[: patch.siibra_mask.shape[0], : patch.siibra_mask.shape[1]] = (
        patch.siibra_mask
    )

    new_siibra_depth_maps = np.zeros(
        (desired_size[0], desired_size[1], patch.siibra_depth_maps.shape[2])
    )
    new_siibra_depth_maps[
        : patch.siibra_depth_maps.shape[0], : patch.siibra_depth_maps.shape[1], :
    ] = patch.siibra_depth_maps

    new_matched_image = np.zeros(desired_size)
    new_matched_image[
        : patch.matched_image.shape[0], : patch.matched_image.shape[1]
    ] = patch.matched_image

    return cort.CorticalPatch(
        image=pad_image(patch.image, desired_size),
        mask=new_mask,
        siibra_image=new_siibra_image,
        siibra_mask=new_siibra_mask,
        matched_image=new_matched_image,
        inplane_resolution_micron=patch.inplane_resolution_micron,
        section_thickness_micron=patch.section_thickness_micron,
        brain_id=patch.brain_id,
        section_id=patch.section_id,
        patch_id=patch.patch_id,
        brain_area=patch.brain_area,
        x=patch.x,
        y=patch.y,
        z=patch.z,
        region_probs=patch.region_probs,
        borders=new_borders,
        depth_maps=new_depth_maps,
        siibra_depth_maps=new_siibra_depth_maps,
    )


def pad_patches(patches: list[cort.CorticalPatch], desired_size: tuple[int, int]):
    return [pad_patch(patch, desired_size) for patch in patches]


def pad_image(img: np.ndarray, desired_size: tuple[int, int]) -> np.ndarray:
    """Pad an image to a desired size."""
    if img.shape[0] > desired_size[0] or img.shape[1] > desired_size[1]:
        raise ValueError(
            f"Image is larger than desired size: {img.shape} > {desired_size}"
        )

    new_img = np.zeros(desired_size)
    new_img[: img.shape[0], : img.shape[1]] = img

    return new_img


def pad_mask(
    mask: np.ndarray, desired_size: tuple[int, int] | tuple[int, int, int]
) -> np.ndarray:
    """Pad a mask to a desired size."""
    if mask.shape[0] > desired_size[0] or mask.shape[1] > desired_size[1]:
        raise ValueError(
            f"Mask is larger than desired size: {mask.shape} > {desired_size}"
        )

    new_mask = np.full(desired_size, cort.constants.PADDING_MASK_VALUE)
    new_mask[: mask.shape[0], : mask.shape[1]] = mask

    return new_mask
