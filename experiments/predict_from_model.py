import json
import os

import numpy as np
import torch

import datasets
import experiments
import nnet.protocols
import nnet.models
import nnet.training


TEMP_MODEL = "models/unet_temp.pth"

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


def predict_from_model(
    fold: int,
    fold_data: datasets.Fold,
    outputs_path: str,
    model: nnet.protocols.SegModelProtocol | None = None,
):
    if model is None:
        model = nnet.models.SemantSegUNetModel.restore(TEMP_MODEL)

    print("Predicting test set...")
    base_results = get_seg_output_for_loader(model, fold_data.test_dataloader)
    print("Saving test set results...")
    os.makedirs(os.path.join(outputs_path, "test"), exist_ok=True)
    base_results.save(os.path.join(outputs_path, "test"))

    print("Calculating stats...")
    for corner_select in [ALL, MAIN_ONLY, CORNER_ONLY]:
        calc_exp_stats(base_results, os.path.join(outputs_path, "test"), corner_select)

    print("Predicting on Siibra patches test set...")
    siibra_results = get_seg_output_for_loader(model, fold_data.siibra_test_dataloader)
    print("Saving Siibra patches test set results...")
    os.makedirs(os.path.join(outputs_path, "siibra_test"), exist_ok=True)
    siibra_results.save(os.path.join(outputs_path, "siibra_test"))

    print("Calculating stats...")
    for corner_select in [ALL, MAIN_ONLY, CORNER_ONLY]:
        calc_exp_stats(
            siibra_results, os.path.join(outputs_path, "siibra_test"), corner_select
        )

    print("Done!")


def get_seg_output_for_loader(
    model: nnet.protocols.SegModelProtocol,
    loader: torch.utils.data.DataLoader,
    transpose: bool = False,
) -> datasets.datatypes.PatchDataItems:
    """Get the segmentation output for a data loader."""
    seg_outputs = []

    for batch in loader:
        batch_info, batch_inputs, batch_gt = batch
        batch_info = datasets.datatypes.PatchInfos(
            brain_area=batch_info[0],
            section_id=batch_info[1].detach().cpu().numpy().tolist(),
            patch_id=batch_info[2].detach().cpu().numpy().tolist(),
            fold=batch_info[3].detach().cpu().numpy().tolist(),
            is_corner_patch=batch_info[4].detach().cpu().numpy().tolist(),
        )
        batch_inputs = datasets.datatypes.SegInputs(*batch_inputs)
        batch_gt = datasets.datatypes.SegGroundTruths(*batch_gt)
        if transpose:
            batch_inputs.input_images = batch_inputs.input_images.transpose(2, 3)

        with torch.no_grad():
            batch_output = model.predict(batch_inputs)

        batch_data_items = (
            datasets.datatypes.PatchDataItems(
                patch_info=batch_info,
                data_inputs=batch_inputs,
                ground_truths=batch_gt,
                predictions=batch_output,
            )
            .detach()
            .cpu()
        )

        seg_outputs.append(batch_data_items)

    seg_output = seg_outputs[0]
    for i in range(1, len(seg_outputs)):
        seg_output += seg_outputs[i]

    return seg_output


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
