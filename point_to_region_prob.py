import os

import numpy as np
import siibra
import nibabel as nib
from tqdm import tqdm

import cort.load

CORT_BASE_DIR = "data/all_cort_patches"


def main():
    cort_folders = cort.load.find_all_patches(CORT_BASE_DIR)

    points = []
    for folder in cort_folders:
        image = nib.load(os.path.join(folder, "image.nii.gz"))
        vol = siibra.volumes.volume.from_nifti(image, space="bigbrain", name="point")
        points.append(vol.get_boundingbox().center)

    julich_pmaps = siibra.get_map(
        parcellation="julich 2.9", space="mni152", maptype=siibra.MapType.STATISTICAL
    )

    regions_probs = []

    for point in tqdm(points, desc="Assigning points"):
        with siibra.QUIET:
            assignments = julich_pmaps.assign(point)
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

    all_regions = list(
        set(
            [region for region_probs in regions_probs for region in region_probs.keys()]
        )
    )
    all_regions.sort()
    print("NUM REGIONS:", len(all_regions))

    probs = np.zeros((len(points), len(all_regions)))

    for i, region_probs in enumerate(regions_probs):
        for j, region in enumerate(all_regions):
            probs[i, j] = region_probs.get(region, 0)
        # probs[i] = np.exp(probs[i]) / np.sum(np.exp(probs[i]))
        probs[i] = probs[i] / np.sum(probs[i])

    for prob, folder in zip(probs, cort_folders):
        filename = os.path.join(folder, "region_probs.npy")
        np.save(filename, prob)


if __name__ == "__main__":
    main()
