import itertools
import json
import os

import numpy as np
from tqdm import tqdm

import datasets
import experiments

MACRO_F1 = "macro_f1"
MACRO_F1_DENOISED = "macro_f1_denoised"
F1_BY_CLASS = "f1_by_class"
F1_BY_CLASS_DENOISED = "f1_by_class_denoised"
F1_BY_AREA = "f1_by_area"
F1_BY_AREA_DENOISED = "f1_by_area_denoised"
F1_BEST_CUTOFF = "f1_best_cutoff"
F1_WORST_CUTOFF = "f1_worst_cutoff"
PER_PIXEL_ACCURACY = "per_pixel_accuracy"
PER_PIXEL_ACCURACY_DENOISED = "per_pixel_accuracy_denoised"

ALL = "all"
MAIN_ONLY = "main_only"
CORNER_ONLY = "corner_only"

ORIGINAL = "original"
SIIBRA = "siibra"


def calc_exp_stats(folder_path: str, corner_select: str):
    all_results = datasets.datatypes.PatchDataItems.load(folder_path)

    if corner_select == MAIN_ONLY:
        results = all_results.filter(lambda x: not x.is_corner_patch)
    elif corner_select == CORNER_ONLY:
        results = all_results.filter(lambda x: x.is_corner_patch)
    else:
        results = all_results

    macro_f1 = experiments.calc_macro_f1(results)
    f1_by_class = experiments.calc_f1_by_class(results)
    macro_f1_by_area = experiments.calc_f1_by_brain_area(results)
    f1_best_cutoff = np.quantile(np.array(list(macro_f1_by_area.values())), 0.8)
    f1_worst_cutoff = np.quantile(np.array(list(macro_f1_by_area.values())), 0.2)

    f1_best_cutoff = [
        area for area, f1 in macro_f1_by_area.items() if f1 >= f1_best_cutoff
    ]
    f1_worst_cutoff = [
        area for area, f1 in macro_f1_by_area.items() if f1 <= f1_worst_cutoff
    ]
    per_pixel_accuracy = experiments.calc_per_pixel_accuracy(results)

    if results[0].prediction.denoised_segmentation is not None:
        f1_by_class_denoised = experiments.calc_f1_by_class(results, denoised=True)
        macro_f1_denoised = experiments.calc_macro_f1(results, denoised=True)
        macro_f1_by_area_denoised = experiments.calc_f1_by_brain_area(
            results, denoised=True
        )
        per_pixel_accuracy_denoised = experiments.calc_per_pixel_accuracy(
            results, denoised=True
        )
    else:
        f1_by_class_denoised = {area: 0.0 for area in f1_by_class}
        macro_f1_denoised = 0.0
        macro_f1_by_area_denoised = {area: 0.0 for area in macro_f1_by_area}
        per_pixel_accuracy_denoised = 0.0

    output_dict = {
        MACRO_F1: macro_f1,
        MACRO_F1_DENOISED: macro_f1_denoised,
        F1_BY_CLASS: f1_by_class,
        F1_BY_CLASS_DENOISED: f1_by_class_denoised,
        F1_BY_AREA: macro_f1_by_area,
        F1_BY_AREA_DENOISED: macro_f1_by_area_denoised,
        F1_BEST_CUTOFF: f1_best_cutoff,
        F1_WORST_CUTOFF: f1_worst_cutoff,
        PER_PIXEL_ACCURACY: per_pixel_accuracy,
        PER_PIXEL_ACCURACY_DENOISED: per_pixel_accuracy_denoised,
    }

    json.dump(
        output_dict,
        open(f"{folder_path}/{corner_select}_stats.json", "w"),
        indent=4,
        cls=MyEncoder,
    )


def main():
    folders = os.listdir("experiments/results")
    folders = [f for f in folders if os.path.isdir(f"experiments/results/{f}")]
    folders = [f for f in folders if os.path.exists(f"experiments/results/{f}/test/")]

    for folder, kind in tqdm(
        itertools.product(folders, [ORIGINAL, SIIBRA]), total=len(folders) * 2
    ):
        inner_folder = "test" if kind == ORIGINAL else "siibra_test"
        calc_exp_stats(f"experiments/results/{folder}/{inner_folder}", MAIN_ONLY)
        calc_exp_stats(f"experiments/results/{folder}/{inner_folder}", CORNER_ONLY)
        calc_exp_stats(f"experiments/results/{folder}/{inner_folder}", ALL)


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


if __name__ == "__main__":
    main()
