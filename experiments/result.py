import json
from dataclasses import dataclass
import os

import numpy as np
from tqdm import tqdm
import pandas as pd

import cort.manip

IMAGE = "image"
GROUND_TRUTH = "ground_truth"
PREDICTION = "prediction"
IMAGE_PADDED = "image_padded"
GROUND_TRUTH_PADDED = "ground_truth_padded"
PREDICTION_PADDED = "prediction_padded"
FOLD = "fold_data"
BRAIN_AREA = "brain_area"
SECTION_ID = "section_id"
PATCH_ID = "patch_id"
DEPTH_PREDICTION = "depth_prediction"
DEPTH_GROUND_TRUTH = "depth_ground_truth"
DEPTH_PREDICTION_PADDED = "depth_prediction_padded"
DEPTH_GROUND_TRUTH_PADDED = "depth_ground_truth_padded"
PADDED_SUFFIX = "_padded"
FULL_RESULT = "full_result"
ACC_PRED = "acc_pred"


@dataclass
class SingleResult:
    image: np.ndarray
    ground_truth: np.ndarray
    prediction: np.ndarray
    image_padded: np.ndarray
    ground_truth_padded: np.ndarray
    prediction_padded: np.ndarray
    fold: int
    brain_area: str
    section_id: int
    patch_id: int
    acc_pred: float
    depth_prediction: np.ndarray | None = None
    depth_ground_truth: np.ndarray | None = None
    depth_prediction_padded: np.ndarray | None = None
    depth_ground_truth_padded: np.ndarray | None = None


class ExperimentResults:
    def __init__(self, results: list[SingleResult]):
        self.brain_areas = set(result.brain_area for result in results)

        self._results = pd.DataFrame(
            {
                IMAGE: [result.image for result in results],
                GROUND_TRUTH: [result.ground_truth for result in results],
                PREDICTION: [result.prediction for result in results],
                IMAGE_PADDED: [result.image_padded for result in results],
                GROUND_TRUTH_PADDED: [result.ground_truth_padded for result in results],
                PREDICTION_PADDED: [result.prediction_padded for result in results],
                FOLD: [result.fold for result in results],
                BRAIN_AREA: [result.brain_area for result in results],
                SECTION_ID: [result.section_id for result in results],
                PATCH_ID: [result.patch_id for result in results],
                DEPTH_PREDICTION: [result.depth_prediction for result in results],
                DEPTH_GROUND_TRUTH: [result.depth_ground_truth for result in results],
                DEPTH_PREDICTION_PADDED: [
                    result.depth_prediction_padded for result in results
                ],
                DEPTH_GROUND_TRUTH_PADDED: [
                    result.depth_ground_truth_padded for result in results
                ],
                FULL_RESULT: results,
                ACC_PRED: [result.acc_pred for result in results],
            }
        )

    def get_example(self, brain_area: str | None = None):
        results = self._results
        if brain_area is not None:
            results = results[results[BRAIN_AREA] == brain_area]

        return results.iloc[np.random.randint(len(results))][FULL_RESULT]

    @classmethod
    def from_folder(
        cls,
        folder_path: str,
        report_progress: bool = False,
        transpose: bool = False,
        siibra: bool = False,
    ):
        return cls(
            load_all_results_from_folder(
                folder_path, report_progress, transpose, siibra
            )
        )

    def sample(self, n: int):
        indices = np.random.choice(len(self), n, replace=False)
        return ExperimentResults([self[i] for i in indices])

    def results_for_brain_area(self, brain_area: str):
        return ExperimentResults(
            [result for result in self if result.brain_area == brain_area]
        )

    def __iter__(self):
        return iter(self._results[FULL_RESULT])

    def __getitem__(self, item):
        return self._results[FULL_RESULT].iloc[item]

    def __len__(self):
        return len(self._results)

    def _get_column(
        self,
        column: str,
        brain_area: str | None = None,
        fold: int | None = None,
        padded: bool = False,
    ):
        results = self._results
        if padded:
            column += PADDED_SUFFIX
        if brain_area is not None:
            results = results[results[BRAIN_AREA] == brain_area]
        if fold is not None:
            results = results[results[FOLD] == fold]
        if padded:
            return np.stack(results[column].to_numpy(), axis=0)
        return results[column].array

    def get_images(
        self,
        brain_area: str | None = None,
        fold: int | None = None,
        padded: bool = False,
    ):
        return self._get_column(IMAGE, brain_area, fold, padded)

    def get_ground_truths(
        self,
        brain_area: str | None = None,
        fold: int | None = None,
        padded: bool = False,
    ):
        return self._get_column(GROUND_TRUTH, brain_area, fold, padded)

    def get_predictions(
        self,
        brain_area: str | None = None,
        fold: int | None = None,
        padded: bool = False,
    ):
        return self._get_column(PREDICTION, brain_area, fold, padded)

    def get_depth_predictions(
        self,
        brain_area: str | None = None,
        fold: int | None = None,
        padded: bool = False,
    ):
        return self._get_column(DEPTH_PREDICTION, brain_area, fold, padded)

    def get_depth_ground_truths(
        self,
        brain_area: str | None = None,
        fold: int | None = None,
        padded: bool = False,
    ):
        return self._get_column(DEPTH_GROUND_TRUTH, brain_area, fold, padded)


def load_all_results_from_folder(
    folder_path: str,
    report_progress: bool = False,
    transpose: bool = False,
    siibra: bool = False,
) -> list[SingleResult]:
    loaded_results = []

    if report_progress:
        iterator = tqdm(os.listdir(folder_path), desc="Loading results...")
    else:
        iterator = os.listdir(folder_path)

    if siibra:
        suffix = "_siibra"
    elif transpose:
        suffix = "_t"
    else:
        suffix = ""

    accuracy_file = json.load(open(f"{folder_path}/test_acc_pred{suffix}.json"))

    for file_name in iterator:
        if file_name.endswith(f"_img_pred{suffix}.npy"):
            base_fp = file_name.replace(f"_img_pred{suffix}.npy", "")
            fp_parts = base_fp.split("_")
            fold = int(fp_parts[-1])
            patch_id = int(fp_parts[-2])
            section_id = int(fp_parts[-3])
            brain_area = "_".join(fp_parts[:-3])

            image_padded = np.load(f"{folder_path}/{base_fp}_input{suffix}.npy")
            ground_truth_padded = np.load(f"{folder_path}/{base_fp}_img_gt{suffix}.npy")
            prediction_padded = np.load(f"{folder_path}/{base_fp}_img_pred{suffix}.npy")

            if os.path.exists(f"{folder_path}/{base_fp}_depth_pred{suffix}.npy"):
                depth_prediction_padded = np.load(
                    f"{folder_path}/{base_fp}_depth_pred{suffix}.npy"
                )
                depth_ground_truth_padded = np.load(
                    f"{folder_path}/{base_fp}_depth_gt{suffix}.npy"
                )

                depth_predictions = []
                depth_gts = []
                for i in range(depth_prediction_padded.shape[0]):
                    depth_predictions.append(
                        cort.manip.unpad_img(
                            depth_prediction_padded[i, :, :], ground_truth_padded
                        )
                    )
                    depth_gts.append(
                        cort.manip.unpad_img(
                            depth_ground_truth_padded[:, :, i], ground_truth_padded
                        )
                    )
                depth_prediction = np.stack(depth_predictions, axis=2)
                depth_ground_truth = np.stack(depth_gts, axis=2)

            else:
                depth_prediction_padded = None
                depth_ground_truth_padded = None
                depth_prediction = None
                depth_ground_truth = None

            image = cort.manip.unpad_img(image_padded, ground_truth_padded)
            prediction = cort.manip.unpad_img(prediction_padded, ground_truth_padded)
            ground_truth = cort.manip.unpad_mask(ground_truth_padded)

            loaded_results.append(
                SingleResult(
                    image=image,
                    ground_truth=ground_truth,
                    prediction=prediction,
                    image_padded=image_padded,
                    ground_truth_padded=ground_truth_padded,
                    prediction_padded=prediction_padded,
                    fold=fold,
                    brain_area=brain_area,
                    section_id=section_id,
                    patch_id=patch_id,
                    depth_prediction=depth_prediction,
                    depth_ground_truth=depth_ground_truth,
                    depth_prediction_padded=depth_prediction_padded,
                    depth_ground_truth_padded=depth_ground_truth_padded,
                    acc_pred=accuracy_file[base_fp],
                )
            )

    return loaded_results
