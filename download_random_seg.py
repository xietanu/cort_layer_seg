import itertools
import json
import os
import sys

import numpy as np
import siibra
import cv2
from matplotlib import pyplot as plt
from tqdm import tqdm

import cort.display


def download_patches():
    sampler = siibra.experimental.CorticalProfileSampler()
    for hemisphere in ["right"]:  # "left", "right"]:
        print(f"### {hemisphere} ###")
        all_areas = open("data/all_cort_patches/regions.txt", "r").readlines()

        for area in tqdm(all_areas):
            try:
                area = area.strip()
                print(f"\nFinding imgs for {area} {hemisphere}...")
                found = 0

                region_name = area + " " + hemisphere

                parc = siibra.parcellations.get("julich 3")
                region = parc.get_region(region_name)
                parc_map = parc.get_map("mni152", "statistical")
                pmap = parc_map.fetch(region)

                v = siibra.volumes.from_nifti(pmap, space="bigbrain", name=region.name)
                features = siibra.features.get(v, "CellbodyStainedSection")

                idxs = np.arange(len(features))
                np.random.shuffle(idxs)
                features = [features[i] for i in idxs]
                for idx, section in zip(idxs, features):
                    if found >= 10:
                        break
                    try:
                        imgplane = siibra.experimental.Plane3D.from_image(section)

                        lmap = siibra.get_map("cortical layers", space="bigbrain")
                        l4 = lmap.parcellation.get_region("4 " + hemisphere)

                        contour = imgplane.intersect_mesh(lmap.fetch(l4, format="mesh"))

                        points = siibra.locations.from_points(
                            sum(map(list, contour), [])
                        )
                        probs = v.evaluate_points(points)

                        probs = np.array(probs)
                        if np.all(probs < 0.01):
                            continue
                        probs = probs / np.sum(probs)

                        idx = np.random.choice(len(points), p=probs)

                        prof = sampler.query(points[idx])
                        canvas = imgplane.get_enclosing_patch(prof)
                        # canvas.flip()
                        patch = canvas.extract_volume(section, resolution_mm=0.02)

                        space = siibra.spaces["bigbrain"]
                        bigbrain_template = space.get_template()

                        bbox = patch.get_boundingbox().zoom(2.5)

                        min_pnt, max_pnt = bbox

                        if (max_pnt[0] - min_pnt[0]) < 11:
                            min_pnt[0] -= (11 - max_pnt[0] + min_pnt[0]) / 2
                            max_pnt[0] = min_pnt[0] + 11

                        if (max_pnt[2] - min_pnt[2]) < 11:
                            min_pnt[2] -= (11 - max_pnt[2] + min_pnt[2]) / 2
                            max_pnt[2] = min_pnt[2] + 11

                        max_pnt[1] = (max_pnt[1] + min_pnt[1]) / 2
                        min_pnt[1] = max_pnt[1]

                        pnt_set = siibra.PointSet(
                            [min_pnt, max_pnt],
                            space="bigbrain",
                        )

                        bbox = pnt_set.boundingbox

                        layermap = siibra.get_map(
                            space="bigbrain", parcellation="layers"
                        )
                        mask = layermap.fetch(
                            fragment=f"{hemisphere} hemisphere",
                            resolution_mm=-1,
                            voi=bbox,
                        )
                        mask = mask.get_fdata()

                        position = patch.get_boundingbox().center

                        mask = mask.squeeze()

                        y_excess = max(mask.shape[0] - 512, 0)
                        x_excess = max(mask.shape[1] - 512, 0)
                        mask = mask[
                            y_excess // 2 : mask.shape[0] - y_excess // 2,
                            x_excess // 2 : mask.shape[1] - x_excess // 2,
                        ]
                        mask = mask[:512, :512]

                        print(f"MASK SHAPE: {mask.shape}")

                        store_patch(mask, position, area, hemisphere, idx)
                        print(f"STORED PATCH for {area} {hemisphere} {idx}")
                        found += 1

                    except Exception as e:
                        print(e)
                        continue
            except Exception as e:
                print(e)
                continue


def store_patch(mask, pos, area_name, hemisphere, section_n):
    parts = area_name.strip().split(" ")
    if parts[0] == "Area":
        area_name = parts[1]
    else:
        area_name = parts[0]
    coords = np.array(pos)
    prob = get_prob_from_pos(pos)
    region_name = area_name + "_" + hemisphere + "_" + str(section_n)
    os.makedirs(f"data/random_masks/{region_name}", exist_ok=True)
    cv2.imwrite(f"data/random_masks/{region_name}/mask.png", mask)
    np.save(f"data/random_masks/{region_name}/coords.npy", coords)
    np.save(f"data/random_masks/{region_name}/probs.npy", prob)
    info = {
        "area": area_name,
        "hemisphere": hemisphere,
        "section_n": section_n,
    }
    json.dump(info, open(f"data/random_masks/{region_name}/info.json", "w"))


def get_prob_from_pos(point):
    julich_pmaps = siibra.get_map(
        parcellation="julich 2.9", space="mni152", maptype=siibra.MapType.STATISTICAL
    )

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

    region_probs = cur_region_probs

    with open("data/all_cort_patches/regions.txt", "r") as f:
        all_regions = f.readlines()
        all_regions = [region.strip() for region in all_regions]

    probs = np.zeros(len(all_regions))

    for j, region in enumerate(all_regions):
        probs[j] = region_probs.get(region, 0)
    # probs[i] = np.exp(probs[i]) / np.sum(np.exp(probs[i]))
    probs = probs / np.sum(probs)

    return probs


if __name__ == "__main__":
    download_patches()
