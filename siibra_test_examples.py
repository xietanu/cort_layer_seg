import os

import numpy as np
import siibra
from tqdm import tqdm

HEMISPHERES = ["left", "right"]
PATCH_DATA_FP = "data/all_cort_patches/"


def fetch_examples():
    dirs = os.listdir(PATCH_DATA_FP)
    areas = []
    for d in dirs:
        parts = d.split("_")
        areas.append(parts[-1])
    for area in tqdm(areas):
        for hemisphere in HEMISPHERES:
            with siibra.QUIET:
                parc = siibra.parcellations.get("julich 3")
                try:
                    region = parc.get_region(area + " " + hemisphere)
                    parc_map = parc.get_map("mni152", "statistical")
                    pmap = parc_map.fetch(region)
                    v = siibra.volumes.from_nifti(
                        pmap, space="bigbrain", name=region.name
                    )
                    features = siibra.features.get(v, "CellBodyStainedSection")
                    section = np.random.choice(features)

                    imgplane = siibra.experimental.Plane3D.from_image(section)
                    lmap = siibra.get_map("cortical layers", space="bigbrain")
                    l4 = lmap.parcellation.get_region("4 " + hemisphere)
                    contour = imgplane.intersect_mesh(lmap.fetch(l4, format="mesh"))

                    points = siibra.locations.from_points(sum(map(list, contour), []))
                    probs = v.evaluate_points(points)
                    sampler = siibra.experimental.CorticalProfileSampler()

                    prof = sampler.query(points[probs.argmax()])
                    canvas = imgplane.get_enclosing_patch(prof)
                    canvas.flip()

                    patch = canvas.extract_volume(section, resolution_mm=0.02)

                    img = patch.fetch().get_fdata()
                except:
                    continue

                np.save(f"data/bb_patches/{area}_{hemisphere}.npy", img)


if __name__ == "__main__":
    fetch_examples()
