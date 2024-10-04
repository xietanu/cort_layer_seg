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


def find_corner(
    investigated_corners: list[np.ndarray],
    hemisphere: str,
    area: str,
    section_n: int = 0,
    use_existing: bool = False,
):
    area = area.strip()
    region_name = area + " " + hemisphere

    if use_existing and os.path.exists(f"data/corner_imgs/{region_name}"):
        images = [
            cv2.imread(f"data/corner_imgs/{region_name}/{fn}", cv2.IMREAD_GRAYSCALE)
            for fn in os.listdir(f"data/corner_imgs/{region_name}")
            if fn.endswith(".png")
        ]
        probs = np.load(f"data/corner_imgs/{region_name}/probs.npy")
        coords = np.load(f"data/corner_imgs/{region_name}/coords.npy")
        return images, coords, probs

    parc = siibra.parcellations.get("julich 3")
    region = parc.get_region(region_name)
    parc_map = parc.get_map("mni152", "statistical")
    pmap = parc_map.fetch(region)

    v = siibra.volumes.from_nifti(pmap, space="bigbrain", name=region.name)
    features = siibra.features.get(v, "CellbodyStainedSection")
    if section_n >= len(features):
        print("NO MORE SECTIONS")
        return None, None, None, 1
    section = features[section_n]
    print(section_n, "used.")

    imgplane = siibra.experimental.Plane3D.from_image(section)

    lmap = siibra.get_map("cortical layers", space="bigbrain")
    l4 = lmap.parcellation.get_region("2 " + hemisphere)

    contour = imgplane.intersect_mesh(lmap.fetch(l4, format="mesh"))

    points = siibra.locations.from_points(sum(map(list, contour), []))
    probs = v.evaluate_points(points)

    point_dists = [
        np.linalg.norm(np.array(point1) - np.array(point2))
        for point1, point2 in itertools.pairwise(points)
    ]
    point_dists = np.array(point_dists)
    point_dist_avgs = np.array(
        [np.mean(point_dists[i : i + 5]) for i in range(len(point_dists) - 5)]
    )

    print("prob min:", min(probs[: len(point_dist_avgs)]))
    print("prob max:", max(probs[: len(point_dist_avgs)]))

    point_dist_avgs[probs[: len(point_dist_avgs)] < 0.01] = np.inf

    if np.all(np.isinf(point_dist_avgs)):
        print("NO RELEVANT POINTS FOUND")
        return None, None, None, 0

    idx = np.argmin(point_dist_avgs) + 2

    sampler = siibra.experimental.CorticalProfileSampler()

    prof = sampler.query(points[idx])
    canvas = imgplane.get_enclosing_patch(prof)
    # canvas.flip()
    patch = canvas.extract_volume(section, resolution_mm=0.02)

    space = siibra.spaces["bigbrain"]
    bigbrain_template = space.get_template()

    bbox = patch.get_boundingbox().zoom(1.5)

    min_pnt, max_pnt = bbox

    if (max_pnt[0] - min_pnt[0]) < 6:
        min_pnt[0] -= (6 - max_pnt[0] + min_pnt[0]) / 2
        max_pnt[0] = min_pnt[0] + 6

    if (max_pnt[2] - min_pnt[2]) < 6:
        min_pnt[2] -= (6 - max_pnt[2] + min_pnt[2]) / 2
        max_pnt[2] = min_pnt[2] + 6

    pnt_set = siibra.PointSet(
        [min_pnt, max_pnt],
        space="bigbrain",
    )

    bbox = pnt_set.boundingbox

    for pnt in investigated_corners:
        if np.linalg.norm(pnt - np.array(bbox.center)) < 1:
            print("POINT ALREADY INVESTIGATED")
            return None, None, None, 0

    bigbrain_chunk = bigbrain_template.fetch(voi=bbox, resolution_mm=0.02)
    img = bigbrain_chunk.get_fdata()

    layermap = siibra.get_map(space="bigbrain", parcellation="layers")
    mask = layermap.fetch(
        fragment=f"{hemisphere} hemisphere", resolution_mm=-1, voi=bbox
    )
    mask = mask.get_fdata()

    position = patch.get_boundingbox().center

    img = (img - img.min()) / (img.max() - img.min()) * 255

    x_excess = max(0, img.shape[1] - 256)
    y_excess = max(0, img.shape[0] - 256)

    img = img[
        y_excess // 2 : img.shape[0] - y_excess // 2,
        x_excess // 2 : img.shape[1] - x_excess // 2,
    ]
    mask = mask[
        y_excess // 2 : mask.shape[0] - y_excess // 2,
        x_excess // 2 : mask.shape[1] - x_excess // 2,
    ]
    return img.squeeze(), mask.squeeze(), position, 0


def find_all_corners(skip=0):
    if os.path.exists("data/corner_imgs/investigated_corners.npy"):
        investigated_corners = np.load("data/corner_imgs/investigated_corners.npy")
        investigated_corners = [c for c in investigated_corners]
    else:
        investigated_corners = []

    for hemisphere in ["left", "right"]:
        all_areas = open("data/all_cort_patches/regions.txt", "r").readlines()
        if skip > 0:
            skipped = min(skip, len(all_areas))
            all_areas = all_areas[skip:]
            skip = max(0, skip - skipped)
        for area_name in all_areas:
            kept = 0
            print(f"Finding corners for {area_name} {hemisphere}...")
            section_n = 0
            while section_n > -1:
                try:
                    img, mask, pos, next_area = find_corner(
                        investigated_corners, hemisphere, area_name, section_n
                    )
                    section_n += 1
                    if next_area == 1:
                        break
                    if img is None:
                        continue
                    colour_img = cort.display.colour_patch(img, mask)
                    investigated_corners.append(np.array(pos))
                    np.save(
                        "data/corner_imgs/investigated_corners.npy",
                        np.stack(investigated_corners),
                    )
                    plt.imshow(colour_img)
                    plt.show()
                    keep = input("Keep? (y/n): ")
                    if keep.lower() == "y":
                        store_patch(img, mask, pos, area_name, hemisphere, section_n)
                    #    kept += 1
                    # if kept >= 3:
                    #    break

                except Exception as e:
                    print(e)
                    break


def store_patch(img, mask, pos, area_name, hemisphere, section_n):
    parts = area_name.strip().split(" ")
    if parts[0] == "Area":
        area_name = parts[1]
    else:
        area_name = parts[0]
    coords = np.array(pos)
    prob = get_prob_from_pos(pos)
    region_name = area_name + "_" + hemisphere + "_" + str(section_n)
    os.makedirs(f"data/corner_imgs/{region_name}", exist_ok=True)
    cv2.imwrite(f"data/corner_imgs/{region_name}/img.png", img)
    cv2.imwrite(f"data/corner_imgs/{region_name}/mask.png", mask)
    np.save(f"data/corner_imgs/{region_name}/coords.npy", coords)
    np.save(f"data/corner_imgs/{region_name}/probs.npy", prob)
    info = {
        "area": area_name,
        "hemisphere": hemisphere,
        "section_n": section_n,
    }
    json.dump(info, open(f"data/corner_imgs/{region_name}/info.json", "w"))


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
    find_all_corners()
