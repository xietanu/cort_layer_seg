import os
import sys

import numpy as np
import siibra
import cv2
from tqdm import tqdm


def extract_files(
    hemisphere: str,
    area: str,
    section_n: int = 0,
    steps: int = 50,
    step_size: int = 2,
    use_existing: bool = False,
):
    area = area.strip()
    region_name = area + " " + hemisphere
    affine_mats = []

    if use_existing and os.path.exists(f"data/chained_imgs/{region_name}"):
        images = [
            cv2.imread(f"data/chained_imgs/{region_name}/{fn}", cv2.IMREAD_GRAYSCALE)
            for fn in os.listdir(f"data/chained_imgs/{region_name}")
            if fn.endswith(".png")
        ]
        probs = np.load(f"data/chained_imgs/{region_name}/probs.npy")
        coords = np.load(f"data/chained_imgs/{region_name}/coords.npy")
        affine_mats = [
            np.load(f"data/chained_imgs/{region_name}/{fn}")
            for fn in os.listdir(f"data/chained_imgs/{region_name}")
            if fn.endswith("_aff.npy")
        ]
        return images, coords, probs, affine_mats

    parc = siibra.parcellations.get("julich 3")
    region = parc.get_region(region_name)
    parc_map = parc.get_map("mni152", "statistical")
    pmap = parc_map.fetch(region)

    v = siibra.volumes.from_nifti(pmap, space="bigbrain", name=region.name)
    features = siibra.features.get(v, "CellBodyStainedSection")
    section = features[section_n]

    imgplane = siibra.experimental.Plane3D.from_image(section)
    lmap = siibra.get_map("cortical layers", space="bigbrain")
    l4 = lmap.parcellation.get_region("2 " + hemisphere)
    contour = imgplane.intersect_mesh(lmap.fetch(l4, format="mesh"))

    points = siibra.locations.from_points(sum(map(list, contour), []))
    probs = v.evaluate_points(points)
    sampler = siibra.experimental.CorticalProfileSampler()

    idx = probs.argmax() - (steps * step_size) // 2

    imgs = []
    positions = []

    with siibra.QUIET:
        for i in tqdm(range(0, steps * step_size, step_size), desc="Extracting images"):

            old_stdout = sys.stdout
            sys.stdout = open(os.devnull, "w")
            prof = sampler.query(points[idx + i])
            sys.stdout = old_stdout
            canvas = imgplane.get_enclosing_patch(prof)
            patch = canvas.extract_volume(section, resolution_mm=0.02)

            fetched = patch.fetch()
            affine = fetched.affine
            affine_mats.append(affine)

            position = patch.get_boundingbox().center
            positions.append(position)

            img = fetched.get_fdata()
            imgs.append(img)

    fixed_imgs = [img.squeeze() for img in imgs]

    probabilities = get_prob_from_pos(positions)
    coords = np.array([[point[0], point[1], point[2]] for point in positions])

    os.makedirs(f"data/chained_imgs/{region_name}", exist_ok=True)

    np.save(f"data/chained_imgs/{region_name}/coords.npy", coords)
    np.save(f"data/chained_imgs/{region_name}/probs.npy", probabilities)

    for i, (img, affine) in enumerate(zip(fixed_imgs, affine_mats)):
        print(img.shape)
        cv2.imwrite(f"data/chained_imgs/{region_name}/{i}.png", img)
        np.save(f"data/chained_imgs/{region_name}/{i}_aff.npy", affine)

    return fixed_imgs, coords, probabilities, affine_mats


def get_prob_from_pos(pos):
    julich_pmaps = siibra.get_map(
        parcellation="julich 2.9", space="mni152", maptype=siibra.MapType.STATISTICAL
    )

    regions_probs = []

    for point in tqdm(pos, desc="Assigning points"):
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
