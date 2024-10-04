import numpy as np
import siibra


def get_siibra_patch_from_point(
    point: np.ndarray | tuple[float, float, float],
    target_size: tuple[float, float] = (10, 10),
    get_wagstyl_segmentation: bool = True,
):
    if isinstance(point, tuple):
        point = np.array(point)

    space = siibra.spaces["bigbrain"]
    bigbrain_template = space.get_template()
    # Flatten the bounding box to 2D
    max_pnt = point.copy()
    min_pnt = point.copy()

    height_mm = target_size[0]
    width_mm = target_size[1]

    min_pnt[2] -= height_mm / 2
    max_pnt[2] += height_mm / 2

    min_pnt[0] -= width_mm / 2
    max_pnt[0] += width_mm / 2

    pnt_set = siibra.PointSet(
        [min_pnt, max_pnt],
        space="bigbrain",
    )

    bbox = pnt_set.boundingbox

    point = bbox.center

    bigbrain_chunk = bigbrain_template.fetch(voi=bbox, resolution_mm=0.02)

    bigbrain_arr = bigbrain_chunk.get_fdata()

    siibra_img = bigbrain_arr[:, :, :]

    coords = np.array([point[0], point[1], point[2]])

    if not get_wagstyl_segmentation:
        return siibra_img.squeeze(), None, coords

    layermap = siibra.get_map(parcellation="layers", space="bigbrain")

    if min_pnt[0] < 0:
        cort_mask = layermap.fetch(
            voi=bbox, resolution_mm=0.02, fragment="left hemisphere"
        )
    else:
        cort_mask = layermap.fetch(
            voi=bbox, resolution_mm=0.02, fragment="right hemisphere"
        )

    cort_mask_arr = cort_mask.get_fdata()[:, :, :]

    return siibra_img.squeeze(), cort_mask_arr.squeeze(), coords
