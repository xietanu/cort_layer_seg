import os

import cv2
import numpy as np
import siibra
import torch
from tqdm import tqdm

import chain
import chain.border
import datasets
import nnet.models

RESOLUTION = 0.021167
CACHE_FOLDER = "data/big_patch_cache"

IMAGE = "image"
SEGMENTATION = "segmentation"
SEGMENTATION_DENOISED = "new_segmentation_denoised"
WAGSTYL_SEGMENTATION = "wagstyl_segmentation"
CENTROID_MM = "centroid_mm"
SIZE_MM = "size_mm"

LEFT = "left"
RIGHT = "right"
TOP = "top"
BOTTOM = "bottom"


class SubPatch:
    def __init__(
        self,
        image: np.ndarray,
        centroid_mm: np.ndarray,
        borders: dict[str, int],
        probabilities: np.ndarray,
    ):
        self.image = image
        self.centroid_mm = centroid_mm
        self.borders = borders
        self.probabilities = probabilities


class BigPatch:
    def __init__(
        self,
        image: np.ndarray,
        centroid_mm: np.ndarray,
        size_mm: tuple[int, int],
        wagstyl_segmentation: np.ndarray = None,
        segmentation: np.ndarray = None,
        segmentation_denoised: np.ndarray = None,
    ):
        self.image = image
        self.wagstyl_segmentation = wagstyl_segmentation
        self.segmentation = segmentation
        self.segmentation_denoised = segmentation_denoised
        self.centroid_mm = centroid_mm
        self.subpatches = []
        self.size_mm = size_mm

    @classmethod
    def from_point(
        cls,
        centroid_mm: tuple[float, float, float] | np.ndarray,
        size_mm: tuple[int, int],
        get_wagstyl_segmentation: bool = True,
        use_cache: bool = True,
    ):
        if isinstance(centroid_mm, tuple):
            centroid_mm = np.array(centroid_mm)

        cache_name = f"{centroid_mm[0]*10:.0f}_{centroid_mm[1]*10:.0f}_{centroid_mm[2]*10:.0f}_{size_mm[0]:.0f}_{size_mm[1]:.0f}"

        if use_cache and os.path.exists(f"{CACHE_FOLDER}/{cache_name}.npz"):
            data = np.load(f"{CACHE_FOLDER}/{cache_name}.npz")
            return cls(
                image=data[IMAGE],
                centroid_mm=data[CENTROID_MM],
                size_mm=data[SIZE_MM],
                wagstyl_segmentation=(
                    data[WAGSTYL_SEGMENTATION] if WAGSTYL_SEGMENTATION in data else None
                ),
                segmentation=data[SEGMENTATION] if SEGMENTATION in data else None,
                segmentation_denoised=(
                    data[SEGMENTATION_DENOISED]
                    if SEGMENTATION_DENOISED in data
                    else None
                ),
            )

        image, wagstyl_segmentation, _ = chain.get_siibra_patch_from_point(
            centroid_mm, size_mm, get_wagstyl_segmentation
        )

        os.makedirs(CACHE_FOLDER, exist_ok=True)

        new_big_patch = cls(
            image=image,
            centroid_mm=centroid_mm,
            size_mm=size_mm,
            wagstyl_segmentation=wagstyl_segmentation,
        )

        new_big_patch.save_cache()

        return new_big_patch

    def save_cache(self):
        cache_name = f"{self.centroid_mm[0]*10:.0f}_{self.centroid_mm[1]*10:.0f}_{self.centroid_mm[2]*10:.0f}_{self.size_mm[0]:.0f}_{self.size_mm[1]:.0f}"
        kwargs = {
            IMAGE: self.image,
            CENTROID_MM: self.centroid_mm,
            SIZE_MM: self.size_mm,
        }
        if self.wagstyl_segmentation is not None:
            kwargs[WAGSTYL_SEGMENTATION] = self.wagstyl_segmentation
        if self.segmentation is not None:
            kwargs[SEGMENTATION] = self.segmentation
        if self.segmentation_denoised is not None:
            kwargs[SEGMENTATION_DENOISED] = self.segmentation_denoised

        np.savez(f"{CACHE_FOLDER}/{cache_name}.npz", **kwargs)

    def create_subpatches(self, edge_size: int, step_size: int = 15, depth: int = 80):
        self.subpatches = []
        print("Identifying subpatch positions...")
        shrunk_image = cv2.resize(
            self.image,
            None,
            fx=1 / step_size,
            fy=1 / step_size,
            interpolation=cv2.INTER_NEAREST,
        )

        padding_size = (edge_size // 2) // step_size + 1

        sub_patch_coordinates_px = (
            chain.border.find_below_surface_points(
                shrunk_image,
                depth // step_size,
                shrunk_image.max() * 0.98,
                padding_size,
            )
            * step_size
        )

        for r, c in tqdm(sub_patch_coordinates_px, desc="Creating subpatches"):
            centroid_mm = (
                self.centroid_mm
                + np.array(
                    [c - self.image.shape[1] // 2, 0, r - self.image.shape[0] // 2]
                )
                * RESOLUTION
            )

            borders = {
                LEFT: c - edge_size // 2,
                RIGHT: c + edge_size // 2,
                TOP: r - edge_size // 2,
                BOTTOM: r + edge_size // 2,
            }

            probabilities = get_prob_from_pos(centroid_mm)

            sub_patch = SubPatch(
                image=self.image[
                    borders[TOP] : borders[BOTTOM], borders[LEFT] : borders[RIGHT]
                ],
                centroid_mm=centroid_mm,
                borders=borders,
                probabilities=probabilities,
            )

            self.subpatches.append(sub_patch)

    def predict_segmentation(self, models: list):
        full_image_logits = np.zeros((8, self.image.shape[0], self.image.shape[1]))
        full_image_den_logits = np.zeros((8, self.image.shape[0], self.image.shape[1]))

        if not self.subpatches:
            raise ValueError("No subpatches created")

        for model_n, model in enumerate(models):
            if isinstance(model, str):
                model = nnet.models.SemantSegUNetModel.restore(model)
            for sub_patch in tqdm(
                self.subpatches,
                desc=f"Predicting segmentation with model {model_n + 1}",
            ):
                equiv_point = torch.tensor(sub_patch.centroid_mm).unsqueeze(0).float()
                probs = torch.tensor(sub_patch.probabilities).float()

                sub_img, n_rotations = orient_img(sub_patch.image)

                sub_img = (sub_img - sub_img.min()) / (sub_img.max() - sub_img.min())

                sub_img = torch.tensor(sub_img).unsqueeze(0).unsqueeze(0).float()

                pred = model.predict(
                    datasets.datatypes.SegInputs(sub_img, probs, equiv_point)
                )
                pred_logits = pred.logits.detach().cpu().squeeze(0)
                pred_logits = torch.softmax(pred_logits, dim=0).numpy()

                x, y = np.meshgrid(
                    np.linspace(-2, 2, pred_logits.shape[2]),
                    np.linspace(-2, 2, pred_logits.shape[1]),
                )
                d = np.sqrt(x * x + y * y)
                sigma, mu = 1.0, 0.0
                gauss = 1 / np.exp(d)

                pred_logits = pred_logits * gauss[np.newaxis] * pred.accuracy.item()

                pred_logits = np.rot90(pred_logits, -n_rotations, axes=(1, 2)).copy()

                full_image_logits[
                    :,
                    sub_patch.borders[TOP] : sub_patch.borders[BOTTOM],
                    sub_patch.borders[LEFT] : sub_patch.borders[RIGHT],
                ] += pred_logits

                pred_denoised = pred.denoised_logits.detach().cpu().squeeze(0)
                pred_denoised = torch.softmax(pred_denoised, dim=0).numpy()

                pred_denoised = pred_denoised * gauss[np.newaxis] * pred.accuracy.item()
                pred_denoised = np.rot90(
                    pred_denoised, -n_rotations, axes=(1, 2)
                ).copy()

                full_image_den_logits[
                    :,
                    sub_patch.borders[TOP] : sub_patch.borders[BOTTOM],
                    sub_patch.borders[LEFT] : sub_patch.borders[RIGHT],
                ] += pred_denoised

        self.segmentation = np.argmax(full_image_logits, axis=0)
        self.segmentation_denoised = np.argmax(full_image_den_logits, axis=0)


def orient_img(img):
    white_mask = img > 0.95 * img.max()
    left_count = np.sum(white_mask[:, : img.shape[1] // 2])
    right_count = np.sum(white_mask[:, img.shape[1] // 2 :])
    top_count = np.sum(white_mask[: img.shape[0] // 2, :])
    bottom_count = np.sum(white_mask[img.shape[0] // 2 :, :])

    n_rotations = np.argmax([top_count, right_count, bottom_count, left_count])

    img = np.rot90(img, n_rotations).copy()

    return img, n_rotations


def get_prob_from_pos(pos):
    pos = siibra.Point(pos, space="bigbrain")

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
