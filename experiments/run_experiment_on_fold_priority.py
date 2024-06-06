import json
import os

import numpy as np

import experiments
import nnet.training
import datasets

CONFIGS_PATH = "experiments/configs"


def run_experiment_on_fold_priority(
    config_name: str, fold: int, n_epochs: int = 100, save_name: str | None = None
):
    """Run an experiment on a fold_data."""
    config = json.load(open(f"{CONFIGS_PATH}/{config_name}.json"))

    outputs_path = f"experiments/results/{config_name}"

    if not os.path.exists(outputs_path):
        os.makedirs(outputs_path)

    model = experiments.create_model_from_config(config)

    condition = None
    if config["conditional"]:
        condition = datasets.dataset_conditions_lookup[
            config["dataset_config"]["condition"]
        ]

    fold_data = datasets.load_fold_priority(
        fold=fold, condition=condition, use_position=config["positional"]
    )

    nnet.training.train_on_fold_priority(
        model=model,
        fold=fold_data,
        n_epochs=n_epochs,
        save_name=save_name,
    )

    seg_predictions = []
    depth_predictions = []

    print("Predicting on test set...")
    for test_set in fold_data.test_dataloader:
        test_set_inputs, _ = test_set
        output = model.predict(test_set_inputs)
        if isinstance(output, tuple):
            seg_predictions.append(output[0])
            depth_predictions.append(output[1])
        else:
            seg_predictions.append(output)

    seg_predictions = np.concatenate(seg_predictions, axis=0)
    if len(depth_predictions) > 0:
        depth_predictions = np.concatenate(depth_predictions, axis=0)

    if len(depth_predictions) == 0:
        for pred, patch in zip(seg_predictions, fold_data.test_dataset.patches):
            base_fp = f"{outputs_path}/{patch.brain_area}_{patch.section_id}_{patch.patch_id}_{fold}"
            np.save(f"{base_fp}_img_pred.npy", pred)
            np.save(f"{base_fp}_input.npy", patch.image)
            np.save(f"{base_fp}_img_gt.npy", patch.mask)

    else:
        for pred, depth_pred, patch in zip(
            seg_predictions, depth_predictions, fold_data.test_dataset.patches
        ):
            base_fp = f"{outputs_path}/{patch.brain_area}_{patch.section_id}_{patch.patch_id}_{fold}"
            np.save(f"{base_fp}_img_pred.npy", pred)
            np.save(f"{base_fp}_depth_pred.npy", depth_pred)
            np.save(f"{base_fp}_input.npy", patch.image)
            np.save(f"{base_fp}_img_gt.npy", patch.mask)
            np.save(f"{base_fp}_depth_gt.npy", patch.depth_maps)

    for i in range(len(fold_data.train_dataloader.dataset)):
        print(f"{i}: {fold_data.train_dataloader.weights[i]:.3f}")

    print("Done!")
