import json
import os

import numpy as np

import experiments
import nnet.training
import datasets

CONFIGS_PATH = "experiments/configs"


def run_experiment_on_fold(
    config_name: str,
    fold: int,
    n_epochs: int = 100,
    save_name: str | None = None,
    continue_training: bool = False,
):
    """Run an experiment on a fold_data."""
    config = json.load(open(f"{CONFIGS_PATH}/{config_name}.json"))

    outputs_path = f"experiments/results/{config_name}"

    if not os.path.exists(outputs_path):
        os.makedirs(outputs_path)

    model = experiments.create_model_from_config(config)

    if continue_training:
        model.load("models/unet_temp.pth")

    condition = None
    if config["conditional"]:
        condition = datasets.dataset_conditions_lookup[
            config["dataset_config"]["condition"]
        ]

    fold_data = datasets.load_fold(
        fold=fold, condition=condition, use_position=config["positional"], batch_size=4
    )

    train_log = nnet.training.train_on_fold(
        model=model,
        fold=fold_data,
        n_epochs=n_epochs,
        save_name=save_name,
    )

    experiments.predict_from_model(
        fold=fold,
        fold_data=fold_data,
        outputs_path=outputs_path,
        model=model,
        train_log=train_log,
    )
