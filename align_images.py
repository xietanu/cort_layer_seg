import argparse
import os

import cv2
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import siibra
import cort
import cort.load
import cort.depth

from tqdm import tqdm


MAX_FEATURES = 10000
KEEP_PERCENT = 0.05
REPROJECTION_THRESHOLD = 5.0
SCALE = 1.5
IMAGE_FILENAME = "image.nii.gz"
MASK_FILENAME = "layermask.png"

BASIC_ALIGNED_IMG = "basic_aligned_img.png"
BASIC_ALIGNED_MASK = "basic_aligned_mask.png"
BASIC_ALIGNED_DEPTH_MAPS = "basic_aligned_depth_maps.npy"

FULL_ALIGNED_IMG = "full_aligned_img.png"
FULL_ALIGNED_MASK = "full_aligned_mask.png"
FULL_ALIGNED_DEPTH_MAPS = "full_aligned_depth_maps.npy"

SIIBRA_IMG = "siibra_img.png"
EXISTING_CORT_LAYERS = "exist_cort_layers.png"

PATCH_FOLDER = "data/all_cort_patches"
SIIBRA_FILE = "siibra_img.png"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force_download", action="store_true")

    patch_folders = cort.load.find_all_patches(PATCH_FOLDER)
    print("Matching patches to SIIBRA:")
    iterator = tqdm(patch_folders)
    for patch_folder in iterator:
        iterator.set_description(f"Processing {patch_folder}")
        match_patch_to_siibra(patch_folder)
    print("Done!")


def match_patch_to_siibra(patch_folder: str, force_download: bool = False):
    patch_image, mask, patch_data = load_image_and_mask(patch_folder)

    borders = cort.depth.find_borders(mask)
    depth_maps = cort.depth.map_distance_from_border(mask, borders)

    img_affine = calculate_image_affine_transformation(patch_data)

    siibra_img, exist_cort_layers = get_siibra_patch(
        patch_data, patch_folder, force_download
    )

    center_affine = np.eye(3)
    center_affine[:2, 2] = -np.array(patch_image.shape[::-1]) / 2

    recentering_mat = np.eye(3)
    recentering_mat[:2, 2] = np.array(siibra_img.shape[::-1]) / 2

    alighment_affine = recentering_mat @ img_affine @ center_affine

    basic_alignment_img = (
        cv2.warpPerspective(
            patch_image,
            alighment_affine,
            siibra_img.shape[::-1],
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        * 255
    )
    basic_alignment_mask = cv2.warpPerspective(
        mask,
        alighment_affine,
        siibra_img.shape[::-1],
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=cort.constants.PADDING_MASK_VALUE,
        flags=cv2.INTER_NEAREST,
    )

    layers = []
    for i in range(depth_maps.shape[2]):
        layer = depth_maps[:, :, i]
        basic_alignment_depth_maps = cv2.warpPerspective(
            layer,
            alighment_affine,
            siibra_img.shape[::-1],
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
            flags=cv2.INTER_NEAREST,
        )
        layers.append(basic_alignment_depth_maps)
    basic_alignment_depth_maps = np.stack(layers, axis=2)

    patch_to_match = (basic_alignment_img).astype(np.uint8)

    patch_to_match_scaled = patch_to_match  # (
    #    (patch_to_match - np.min(patch_to_match))
    #    / (np.max(patch_to_match) - np.min(patch_to_match))
    #    * 255
    # )
    siibra_img_scaled = siibra_img  # (
    #    (siibra_img - np.min(siibra_img))
    #    / (np.max(siibra_img) - np.min(siibra_img))
    #    * 255
    # )

    large_patch = cv2.resize(
        patch_to_match_scaled, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_CUBIC
    ).astype(np.uint8)
    large_siibra = cv2.resize(
        siibra_img_scaled, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_CUBIC
    ).astype(np.uint8)
    large_mask = cv2.resize(
        basic_alignment_mask, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_NEAREST
    ).astype(np.uint8)
    large_depth_maps = cv2.resize(
        basic_alignment_depth_maps,
        None,
        fx=SCALE,
        fy=SCALE,
        interpolation=cv2.INTER_NEAREST,
    )
    # large_patch_blur = cv2.GaussianBlur(large_patch, (7, 7), 0)
    # large_siibra_blur = cv2.GaussianBlur(large_siibra, (7, 7), 0)

    orb = cv2.ORB_create(MAX_FEATURES)
    (kpsA, descsA) = orb.detectAndCompute(large_patch, None)
    (kpsB, descsB) = orb.detectAndCompute(large_siibra, None)

    if len(kpsA) == 0 or len(kpsB) == 0:
        no_points = []
        if len(kpsA) == 0:
            no_points.append("patch")
        if len(kpsB) == 0:
            no_points.append("siibra")
        raise ValueError(f"No keypoints found in {', '.join(no_points)} image")

    # match the features
    method = cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING
    matcher = cv2.DescriptorMatcher_create(method)
    matches = matcher.match(descsA, descsB, None)

    matches = sorted(matches, key=lambda x: x.distance)
    # keep only the top matches

    keep = max(int(len(matches) * KEEP_PERCENT), 100)
    matches = matches[:keep]

    ptsA = np.zeros((len(matches), 2), dtype="float")
    ptsB = np.zeros((len(matches), 2), dtype="float")
    # loop over the top matches
    for i, m in enumerate(matches):
        # indicate that the two keypoints in the respective images
        # map to each other
        ptsA[i] = kpsA[m.queryIdx].pt
        ptsB[i] = kpsB[m.trainIdx].pt

    (H, _) = cv2.estimateAffine2D(
        ptsA,
        ptsB,
        method=cv2.RANSAC,
        ransacReprojThreshold=REPROJECTION_THRESHOLD,
        confidence=0.99,
    )
    # use the homography matrix to align the images

    if H is None:
        raise ValueError("No homography found")

    (h, w) = large_siibra.shape[:2]
    large_full_alignment_img = cv2.warpAffine(
        large_patch,
        H,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    large_full_alignment_mask = cv2.warpAffine(
        large_mask,
        H,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=cort.constants.PADDING_MASK_VALUE,
    )
    layers = []
    for i in range(depth_maps.shape[2]):
        layer = large_depth_maps[:, :, i]
        large_aligned_depth_maps = cv2.warpAffine(
            layer,
            H,
            (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        layers.append(large_aligned_depth_maps)
    large_aligned_depth_maps = np.stack(layers, axis=2)

    full_alignment_img = cv2.resize(
        large_full_alignment_img,
        None,
        fx=1 / SCALE,
        fy=1 / SCALE,
        interpolation=cv2.INTER_CUBIC,
    )

    full_alignment_mask = cv2.resize(
        large_full_alignment_mask,
        None,
        fx=1 / SCALE,
        fy=1 / SCALE,
        interpolation=cv2.INTER_NEAREST,
    )

    full_aligned_depth_maps = cv2.resize(
        large_aligned_depth_maps,
        None,
        fx=1 / SCALE,
        fy=1 / SCALE,
        interpolation=cv2.INTER_NEAREST,
    )

    cv2.imwrite(os.path.join(patch_folder, BASIC_ALIGNED_IMG), basic_alignment_img)
    cv2.imwrite(os.path.join(patch_folder, BASIC_ALIGNED_MASK), basic_alignment_mask)
    np.save(
        os.path.join(patch_folder, BASIC_ALIGNED_DEPTH_MAPS), basic_alignment_depth_maps
    )

    cv2.imwrite(os.path.join(patch_folder, FULL_ALIGNED_IMG), full_alignment_img)
    cv2.imwrite(os.path.join(patch_folder, FULL_ALIGNED_MASK), full_alignment_mask)
    np.save(
        os.path.join(patch_folder, FULL_ALIGNED_DEPTH_MAPS), full_aligned_depth_maps
    )

    cv2.imwrite(os.path.join(patch_folder, SIIBRA_IMG), siibra_img)
    cv2.imwrite(os.path.join(patch_folder, EXISTING_CORT_LAYERS), exist_cort_layers)


def calculate_image_affine_transformation(patch_data: nib.Nifti1Image):
    resolution = patch_data.header["pixdim"][1:4]
    affine = patch_data.affine

    scale = np.diag(1 / resolution)
    scale[0, 0] = -scale[0, 0]

    trimmed_affine = np.eye(3)
    trimmed_affine[:2, :2] = np.stack([affine[0, :2], affine[2, :2]], axis=0)

    affine = scale @ trimmed_affine

    inv_affine = np.linalg.inv(affine)

    return inv_affine


def get_siibra_patch(patch_data: nib.Nifti1Image, folder: str, force_download: bool):
    if (
        os.path.isfile(os.path.join(folder, SIIBRA_FILE))
        and os.path.isfile(os.path.join(folder, EXISTING_CORT_LAYERS))
        and not force_download
    ):
        return (
            cv2.imread(os.path.join(folder, SIIBRA_FILE), cv2.IMREAD_UNCHANGED),
            cv2.imread(
                os.path.join(folder, EXISTING_CORT_LAYERS), cv2.IMREAD_UNCHANGED
            ),
        )

    space = siibra.spaces["bigbrain"]
    layermap = siibra.get_map(parcellation="layers", space="bigbrain")
    bigbrain_template = space.get_template()

    vol = siibra.volumes.volume.from_nifti(patch_data, space="bigbrain", name="orig")

    bbox = vol.get_boundingbox()

    # Flatten the bounding box to 2D
    min_pnt, max_pnt = bbox
    max_pnt = np.array(max_pnt)
    max_pnt[1] = min_pnt[1]
    min_pnt = np.array(min_pnt)

    if (max_pnt[0] - min_pnt[0]) < 2:
        min_pnt[0] -= (2 - max_pnt[0] + min_pnt[0]) / 2
        max_pnt[0] = min_pnt[0] + 2

    if (max_pnt[2] - min_pnt[2]) < 2:
        min_pnt[2] -= (2 - max_pnt[2] + min_pnt[2]) / 2
        max_pnt[2] = min_pnt[2] + 2

    pnt_set = siibra.PointSet(
        [min_pnt, max_pnt],
        space="bigbrain",
    )

    bbox = pnt_set.boundingbox
    bbox = bbox.zoom(1.75)

    bigbrain_chunk = bigbrain_template.fetch(voi=bbox, resolution_mm=0.02)

    bigbrain_arr = bigbrain_chunk.get_fdata()

    siibra_img = bigbrain_arr[:, 0, :].T

    if min_pnt[0] < 0:
        cort_mask = layermap.fetch(
            voi=bbox, resolution_mm=0.02, fragment="left hemisphere"
        )
    else:
        cort_mask = layermap.fetch(
            voi=bbox, resolution_mm=0.02, fragment="right hemisphere"
        )

    cort_mask_arr = cort_mask.get_fdata()

    cort_mask_img = cort_mask_arr[:, 0, :].T

    return siibra_img, cort_mask_img


def load_image_and_mask(patch_folder: str):
    image = nib.load(os.path.join(patch_folder, IMAGE_FILENAME))

    layer = image.get_fdata()
    layer = (layer - np.min(layer)) / (np.max(layer) - np.min(layer))

    mask = cv2.imread(os.path.join(patch_folder, MASK_FILENAME), cv2.IMREAD_UNCHANGED)

    dscale_layer, dscale_mask = downscale_image_and_mask(layer, mask, 20)

    dscale_layer = dscale_layer[:, 1:]
    dscale_mask = dscale_mask[:, 1:]

    mask_top_half = dscale_mask[: dscale_mask.shape[0] // 2, :]
    mask_top_half[mask_top_half == 7] = 0
    dscale_mask[: dscale_mask.shape[0] // 2, :] = mask_top_half

    return dscale_layer, dscale_mask, image


def downscale_image_and_mask(layer: np.ndarray, mask: np.ndarray, downscale: float):
    layer = cv2.resize(
        layer, None, fx=1 / downscale, fy=1 / downscale, interpolation=cv2.INTER_AREA
    )
    mask = cv2.resize(
        mask, None, fx=1 / downscale, fy=1 / downscale, interpolation=cv2.INTER_NEAREST
    )

    return layer, mask


if __name__ == "__main__":
    main()
