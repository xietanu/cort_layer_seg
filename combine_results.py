import argparse
import json
import os

import numpy as np
import torch

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

ALL = "all"
MAIN_ONLY = "main_only"
CORNER_ONLY = "corner_only"

ORIGINAL = "original"
SIIBRA = "siibra"


def main():
    args = argparse.ArgumentParser()

    args.add_argument(
        "models",
        metavar="M",
        type=str,
        nargs="+",
        help="The models to combine",
    )

    args.add_argument("--output", type=str)

    args = args.parse_args()

    combined = None
    combined_siibra = None

    accuracies = []
    siibra_accuracies = []

    for model in args.models:
        if combined is None:
            combined = datasets.datatypes.PatchDataItems.load(
                f"experiments/results/{model}/test"
            )
            combined_siibra = datasets.datatypes.PatchDataItems.load(
                f"experiments/results/{model}/siibra_test"
            )
            combined.predictions.logits = torch.softmax(combined.predictions.logits, 1)
            combined.predictions.denoised_logits = torch.softmax(
                combined.predictions.denoised_logits, 1
            )
            combined_siibra.predictions.logits = torch.softmax(
                combined_siibra.predictions.logits, 1
            )
            combined_siibra.predictions.denoised_logits = torch.softmax(
                combined_siibra.predictions.denoised_logits, 1
            )

            accuracies.append(combined.predictions.accuracy)
            siibra_accuracies.append(combined_siibra.predictions.accuracy)

        else:
            new_results = datasets.datatypes.PatchDataItems.load(
                f"experiments/results/{model}/test"
            )
            new_results.predictions.logits = torch.softmax(
                new_results.predictions.logits, 1
            )
            new_results.predictions.denoised_logits = torch.softmax(
                new_results.predictions.denoised_logits, 1
            )
            combined.predictions.logits += new_results.predictions.logits
            combined.predictions.denoised_logits += (
                new_results.predictions.denoised_logits
            )
            combined.predictions.segmentation = torch.argmax(
                combined.predictions.logits, dim=1
            ).unsqueeze(1)
            combined.predictions.denoised_segementation = torch.argmax(
                combined.predictions.denoised_logits, dim=1
            ).unsqueeze(1)

            new_siibra_results = datasets.datatypes.PatchDataItems.load(
                f"experiments/results/{model}/siibra_test"
            )
            combined_siibra.predictions.logits = torch.softmax(
                combined_siibra.predictions.logits, 1
            )
            combined_siibra.predictions.denoised_logits = torch.softmax(
                combined_siibra.predictions.denoised_logits, 1
            )
            combined_siibra.predictions.logits += new_siibra_results.predictions.logits
            combined_siibra.predictions.denoised_logits += (
                new_siibra_results.predictions.denoised_logits
            )
            combined_siibra.predictions.segmentation = torch.argmax(
                combined_siibra.predictions.logits, dim=1
            ).unsqueeze(1)
            combined_siibra.predictions.denoised_segementation = torch.argmax(
                combined_siibra.predictions.denoised_logits, dim=1
            ).unsqueeze(1)

            combined.predictions.depth_maps += new_results.predictions.depth_maps
            combined_siibra.predictions.depth_maps += (
                new_siibra_results.predictions.depth_maps
            )

            accuracies.append(new_results.predictions.accuracy)
            siibra_accuracies.append(new_siibra_results.predictions.accuracy)

    combined.predictions.accuracy = torch.stack(accuracies, dim=0).mean(dim=0)
    combined_siibra.predictions.accuracy = torch.stack(siibra_accuracies, dim=0).mean(
        dim=0
    )

    output_folder = f"experiments/results/{args.output}"

    print("Saving combined results...")
    os.makedirs(f"{output_folder}/test", exist_ok=True)
    os.makedirs(f"{output_folder}/siibra_test", exist_ok=True)
    combined.save(f"{output_folder}/test")
    print("Saving combined siibra results...")
    combined_siibra.save(f"{output_folder}/siibra_test")

    print("Calculating stats...")
    for corner_select in [ALL, MAIN_ONLY, CORNER_ONLY]:
        calc_exp_stats(
            combined,
            os.path.join(output_folder, "test"),
            corner_select,
        )
        calc_exp_stats(
            combined_siibra,
            os.path.join(output_folder, "siibra_test"),
            corner_select,
        )


def calc_exp_stats(all_results, folder_path: str, corner_select: str):
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
    ppa = experiments.calc_per_pixel_accuracy(results)

    if results.predictions.denoised_segementation is not None:
        macro_f1_denoised = experiments.calc_macro_f1(results, denoised=True)
        f1_by_class_denoised = experiments.calc_f1_by_class(results, denoised=True)
        macro_f1_by_area_denoised = experiments.calc_f1_by_brain_area(
            results, denoised=True
        )
        ppa_denoised = experiments.calc_per_pixel_accuracy(results, denoised=True)
    else:
        macro_f1_denoised = 0.0
        f1_by_class_denoised = {cls: 0.0 for cls in f1_by_class}
        macro_f1_by_area_denoised = {area: 0.0 for area in macro_f1_by_area}
        ppa_denoised = 0.0

    output_dict = {
        MACRO_F1: macro_f1,
        MACRO_F1_DENOISED: macro_f1_denoised,
        F1_BY_CLASS: f1_by_class,
        F1_BY_CLASS_DENOISED: f1_by_class_denoised,
        F1_BY_AREA: macro_f1_by_area,
        F1_BY_AREA_DENOISED: macro_f1_by_area_denoised,
        F1_BEST_CUTOFF: f1_best_cutoff,
        F1_WORST_CUTOFF: f1_worst_cutoff,
        "per_pixel_accuracy": ppa,
        "per_pixel_accuracy_denoised": ppa_denoised,
    }

    json.dump(
        output_dict,
        open(f"{folder_path}/{corner_select}_stats.json", "w"),
        indent=4,
        cls=MyEncoder,
    )


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
