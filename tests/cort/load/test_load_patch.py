import cv2
import numpy as np

from cort.load.load_patch import (
    load_patch,
    read_image_and_mask,
    downscale_image_and_mask,
)
import cort

EXAMPLE_FOLDER_PATH_1 = "tests/example_data/brain_region/section_a/0001"
EXAMPLE_FOLDER_PATH_2 = "tests/example_data/brain_region/section_a/0002"

EXAMPLE_PATCH_IMAGE_PATH_1 = "tests/example_data/brain_region/section_a/0001/image.png"
EXAMPLE_PATCH_MASK_PATH_1 = (
    "tests/example_data/brain_region/section_a/0001/layermask.png"
)
EXAMPLE_PATCH_IMAGE_PATH_2 = "tests/example_data/brain_region/section_a/0002/image.png"
EXAMPLE_PATCH_MASK_PATH_2 = (
    "tests/example_data/brain_region/section_a/0002/layermask.png"
)

EXPECTED_PATCH_1 = cort.CorticalPatch(
    cv2.imread(EXAMPLE_PATCH_IMAGE_PATH_1, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255 * 2 - 1,
    cv2.imread(EXAMPLE_PATCH_MASK_PATH_1, cv2.IMREAD_GRAYSCALE),
    inplane_resolution_micron=1,
    section_thickness_micron=20,
    brain_id="A24",
    section_id=123,
    patch_id=1,
    brain_area="fake_area",
)

EXPECTED_PATCH_2 = cort.CorticalPatch(
    cv2.imread(EXAMPLE_PATCH_IMAGE_PATH_2, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255 * 2 - 1,
    cv2.imread(EXAMPLE_PATCH_MASK_PATH_2, cv2.IMREAD_GRAYSCALE),
    inplane_resolution_micron=1,
    section_thickness_micron=20,
    brain_id="A24",
    section_id=123,
    patch_id=2,
    brain_area="fake_area",
)


def test_read_image_and_mask():
    layer, mask = read_image_and_mask(EXAMPLE_FOLDER_PATH_1)
    assert layer.shape == (
        3201,
        973,
    ), f"layer shape should be (3021, 973), not {layer.shape}"
    assert mask.shape == (
        3201,
        973,
    ), f"mask shape should be (3021, 973), not {mask.shape}"
    assert layer.dtype == "float32", f"layer dtype should be float32, not {layer.dtype}"
    assert mask.dtype == "uint8", f"mask dtype should be uint8, not {mask.dtype}"
    assert layer.max() <= 1, "layer max should be normalized to 1"
    assert layer.min() >= -1, "layer min should be normalized to -1"

    assert np.allclose(
        layer, EXPECTED_PATCH_1.image
    ), "layer should be the same as the expected patch image"
    assert np.allclose(
        mask, EXPECTED_PATCH_1.mask
    ), "mask should be the same as the expected patch mask"


def test_downscale_image_and_mask_no_rescale():
    layer, mask = downscale_image_and_mask(
        EXPECTED_PATCH_1.image, EXPECTED_PATCH_1.mask, 1
    )
    assert layer.shape == (
        3201,
        973,
    ), f"layer shape should be (3021, 973), not {layer.shape}"
    assert mask.shape == (
        3201,
        973,
    ), f"mask shape should be (3021, 973), not {mask.shape}"
    assert layer.dtype == "float32", f"layer dtype should be float32, not {layer.dtype}"
    assert mask.dtype == "uint8", f"mask dtype should be uint8, not {mask.dtype}"
    assert layer.max() <= 1, "layer max should be normalized to 1"
    assert layer.min() >= -1, "layer min should be normalized to -1"
    assert np.allclose(
        layer, EXPECTED_PATCH_1.image
    ), "layer should be the same as the expected patch image"
    assert np.allclose(
        mask, EXPECTED_PATCH_1.mask
    ), "mask should be the same as the expected patch mask"


def test_downscale_image_and_mask_downscale_10():
    layer, mask = downscale_image_and_mask(
        EXPECTED_PATCH_1.image, EXPECTED_PATCH_1.mask, 10
    )
    assert layer.shape == (
        320,
        97,
    ), f"layer shape should be (320, 97), not {layer.shape}"
    assert mask.shape == (
        320,
        97,
    ), f"mask shape should be (320, 97), not {mask.shape}"
    assert layer.dtype == "float32", f"layer dtype should be float32, not {layer.dtype}"
    assert mask.dtype == "uint8", f"mask dtype should be uint8, not {mask.dtype}"
    assert layer.max() <= 1, "layer max should be normalized to 1"
    assert layer.min() >= -1, "layer min should be normalized to -1"
    
def test_load_patch_patch_1():
    patch = load_patch(EXAMPLE_FOLDER_PATH_1)
    assert np.allclose(
        patch.image, EXPECTED_PATCH_1.image
    ), "image should be the same as the expected patch image"
    assert np.allclose(
        patch.mask, EXPECTED_PATCH_1.mask
    ), "mask should be the same as the expected patch mask"
    assert patch.inplane_resolution_micron == 1, "inplane resolution should be 1"
    assert patch.section_thickness_micron == 20, "section thickness should be 20"
    assert patch.brain_id == "A24", "brain id should be A24"
    assert patch.section_id == 123, "section id should be 123"
    assert patch.patch_id == 1, "patch id should be 1"
    assert patch.brain_area == "fake_area", "brain area should be fake_area"
    
def test_load_patch_patch_2():
    patch = load_patch(EXAMPLE_FOLDER_PATH_2)
    assert np.allclose(
        patch.image, EXPECTED_PATCH_2.image
    ), "image should be the same as the expected patch image"
    assert np.allclose(
        patch.mask, EXPECTED_PATCH_2.mask
    ), "mask should be the same as the expected patch mask"
    assert patch.inplane_resolution_micron == 1, "inplane resolution should be 1"
    assert patch.section_thickness_micron == 20, "section thickness should be 20"
    assert patch.brain_id == "A24", "brain id should be A24"
    assert patch.section_id == 123, "section id should be 123"
    assert patch.patch_id == 2, "patch id should be 2"
    assert patch.brain_area == "fake_area", "brain area should be fake_area"
