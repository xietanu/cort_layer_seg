import numpy as np
import pandas as pd

import datasets
import evaluate
import cort

CLASS_NAMES = [
    "BG",
    "I",
    "II",
    "III",
    "IV",
    "V",
    "VI",
    "WM",
]


def calc_macro_f1(
    results: datasets.datatypes.PatchDataItems, denoised: bool = False
) -> float:

    return evaluate.f1_score(
        (
            results.predictions.segmentation.squeeze(1)
            if not denoised
            else results.predictions.denoised_segementation.squeeze(1)
        ),
        results.ground_truths.segmentation.squeeze(1),
        ignore_index=cort.constants.PADDING_MASK_VALUE,
        n_classes=8,
    )


def calc_f1_by_class(
    results: datasets.datatypes.PatchDataItems, denoised: bool = False
) -> dict[str, float]:

    scores = evaluate.f1_score(
        (
            results.predictions.segmentation.squeeze(1)
            if not denoised
            else results.predictions.denoised_segementation.squeeze(1)
        ),
        results.ground_truths.segmentation.squeeze(1),
        ignore_index=cort.constants.PADDING_MASK_VALUE,
        n_classes=8,
        average_over_classes=False,
    )
    return {CLASS_NAMES[i]: scores[i] for i in range(8)}  # type: ignore


def calc_f1_by_brain_area(
    results: datasets.datatypes.PatchDataItems, denoised: bool = False
) -> dict[str, float]:

    brain_areas = results.brain_areas

    f1_scores = pd.Series(
        {
            brain_area: evaluate.f1_score(
                (
                    results.predictions.segmentation[
                        np.array(results.patch_info.brain_area) == brain_area
                    ].squeeze(1)
                    if not denoised
                    else results.predictions.denoised_segementation[
                        np.array(results.patch_info.brain_area) == brain_area
                    ].squeeze(1)
                ),
                results.ground_truths.segmentation[
                    np.array(results.patch_info.brain_area) == brain_area
                ].squeeze(1),
                ignore_index=cort.constants.PADDING_MASK_VALUE,
                n_classes=8,
            )
            for brain_area in brain_areas
        }
    )

    f1_scores.sort_values(inplace=True, ascending=False)

    return f1_scores.to_dict()


def calc_f1_by_brain_area_and_class(
    results: datasets.datatypes.PatchDataItems, denoised: bool = False
) -> pd.DataFrame:

    brain_areas = results.brain_areas

    ba_dict = {
        brain_area: evaluate.f1_score(
            (
                results.predictions.segmentation[
                    np.array(results.patch_info.brain_area) == brain_area
                ].squeeze(1)
                if not denoised
                else results.predictions.denoised_segementation[
                    np.array(results.patch_info.brain_area) == brain_area
                ].squeeze(1)
            ),
            results.ground_truths.segmentation[
                np.array(results.patch_info.brain_area) == brain_area
            ].squeeze(1),
            ignore_index=cort.constants.PADDING_MASK_VALUE,
            n_classes=8,
            average_over_classes=False,
        )
        for brain_area in brain_areas
    }
    ba_dict["Overall"] = evaluate.f1_score(
        (
            results.predictions.segmentation.squeeze(1)
            if not denoised
            else results.predictions.denoised_segementation.squeeze(1)
        ),
        results.ground_truths.segmentation.squeeze(1),
        ignore_index=cort.constants.PADDING_MASK_VALUE,
        n_classes=8,
        average_over_classes=False,
    )

    f1_scores = pd.DataFrame(
        ba_dict,
        index=CLASS_NAMES,
    )

    return f1_scores
