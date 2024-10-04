import numpy as np
import nibabel as nib
import siibra


def get_siibra_volume(
    patch_data: nib.Nifti1Image,
    offset: tuple[float, float, float] = (0, 0, 0),
    rot: int = 0,
    rot_axis: tuple[int, int] = (0, 1),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    vol = siibra.volumes.volume.from_nifti(patch_data, space="bigbrain", name="orig")

    bbox = vol.get_boundingbox()

    min_pnt, max_pnt = bbox

    min_pnt = np.array(min_pnt) + np.array(offset)
    max_pnt = np.array(max_pnt) + np.array(offset)

    return get_siibra_volume_from_points(min_pnt, max_pnt, rot, rot_axis)

def get_siibra_volume_from_points(
    min_pnt: np.ndarray,
    max_pnt: np.ndarray = None,
    rot: int = 0,
    rot_axis: tuple[int, int] = (0, 1),
):
    space = siibra.spaces["bigbrain"]
    bigbrain_template = space.get_template()
    # Flatten the bounding box to 2D
    if max_pnt is None:
        max_pnt = min_pnt.copy()

    max_pnt = np.array(max_pnt)
    max_pnt[1] = min_pnt[1] + 2
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
    bbox = bbox.zoom(2)

    point = bbox.center

    bigbrain_chunk = bigbrain_template.fetch(voi=bbox, resolution_mm=0.02)

    bigbrain_arr = bigbrain_chunk.get_fdata()

    siibra_img = bigbrain_arr[:, :, :].transpose(1, 2, 0)

    coords = np.array([point[0], point[1], point[2]])
    probs = get_prob_from_pos(point)

    rot_img = np.rot90(siibra_img, axes=rot_axis, k=rot)

    x_dim = rot_img.shape[0]
    y_dim = rot_img.shape[1]
    z_dim = rot_img.shape[2]

    x_excess = max(0, x_dim - 256)
    y_excess = max(0, y_dim - 256)
    z_excess = max(0, z_dim - 256)

    rot_img = rot_img[
        x_excess // 2 : x_dim - (x_excess - x_excess // 2),
        y_excess // 2 : y_dim - (y_excess - y_excess // 2),
        z_excess // 2 : z_dim - (z_excess - z_excess // 2),
    ]

    rot_img = rot_img[:256, :256, :256]

    layermap = siibra.get_map(parcellation="layers", space="bigbrain")

    if min_pnt[0] < 0:
        cort_mask = layermap.fetch(
            voi=bbox, resolution_mm=0.02, fragment="left hemisphere"
        )
    else:
        cort_mask = layermap.fetch(
            voi=bbox, resolution_mm=0.02, fragment="right hemisphere"
        )

    cort_mask_arr = cort_mask.get_fdata()[:, :, :].transpose(1, 2, 0)
    rot_mask = np.rot90(cort_mask_arr, axes=rot_axis, k=rot)


    rot_mask = rot_mask[
              x_excess // 2: x_dim - (x_excess - x_excess // 2),
              y_excess // 2: y_dim - (y_excess - y_excess // 2),
              z_excess // 2: z_dim - (z_excess - z_excess // 2),
              ]

    rot_mask = rot_mask[:256, :256, :256]

    return rot_img, rot_mask, coords, probs


def get_prob_from_pos(pos):
    julich_pmaps = siibra.get_map(
        parcellation="julich 2.9", space="mni152", maptype=siibra.MapType.STATISTICAL
    )

    regions_probs = []

    with siibra.QUIET:
        assignments = julich_pmaps.assign(pos)
    cur_region_probs = {}
    for i, assignment in assignments.iterrows():
        region = assignment["region"].parent.name
        if region not in cur_region_probs:
            cur_region_probs[region] = []
        cur_region_probs[region].append(assignment["map value"])
    for region, values in cur_region_probs.items():
        cur_region_probs[region] = np.array(values)

    for region, values in cur_region_probs.items():
        cur_region_probs[region] = 1 - np.prod(1 - values)

    regions_probs.append(cur_region_probs)

    with open("data/all_cort_patches/regions.txt", "r") as f:
        all_regions = f.readlines()
        all_regions = [region.strip() for region in all_regions]

    probs = np.zeros((len(pos), len(all_regions)))

    for i, region_probs in enumerate(regions_probs):
        for j, region in enumerate(all_regions):
            probs[i, j] = region_probs.get(region, 0)
        # probs[i] = np.exp(probs[i]) / np.sum(np.exp(probs[i]))
        probs[i] = probs[i] / np.sum(probs[i])

    return probs
