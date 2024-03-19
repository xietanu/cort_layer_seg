import numpy as np

from cort import CorticalPatch

EXAMPLE_PATCH = CorticalPatch(
    image=np.zeros((10, 12)),
    mask=np.zeros((10, 12)),
    inplane_resolution_micron=20,
    section_thickness_micron=30,
    brain_id="D20",
    section_id=456,
    patch_id=42,
    brain_area="not_real_area",
)


def test_patch_name():
    assert (
        EXAMPLE_PATCH.name == "D20-not_real_area-456-42"
    ), f"name should be D20-not_real_area-456-42, not {EXAMPLE_PATCH.name}"


def test_patch_shape():
    assert EXAMPLE_PATCH.shape == (
        10,
        12,
    ), "shape should be (10, 10), not {EXAMPLE_PATCH.shape}"
