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
    n_epochs: int = -1,
    end_after: int = -1,
    save_name: str | None = None,
    continue_training: bool = False,
    use_random: bool = False,
    use_gaussian: bool = False,
    use_synth: bool = False,
    auto_denoise: bool = False,
):
    """Run an experiment on a fold_data."""
    config = json.load(open(f"{CONFIGS_PATH}/{config_name}.json"))

    outputs_path = (
        f"experiments/results/{config_name}"
        if save_name is None
        else f"experiments/results/{save_name}"
    )

    if end_after <= 0 and n_epochs <= 0:
        raise ValueError("Must specify either n_epochs or end_after")

    if not os.path.exists(outputs_path):
        os.makedirs(outputs_path)

    if auto_denoise:
        config["denoise_model_name"] = f"denoise_{save_name}"
        config["accuracy_model_name"] = f"acc_{save_name}"

    model = experiments.create_model_from_config(config)

    if continue_training:
        model.load("models/unet_temp.pth")

    condition = None
    if config["conditional"]:
        condition = datasets.dataset_conditions_lookup[
            config["dataset_config"]["condition"]
        ]

    fold_data = datasets.load_fold(
        fold=fold,
        condition=condition,
        use_position=config["positional"],
        batch_size=4,
        use_random=use_random,
        use_gaussian=use_gaussian,
        use_synth=use_synth,
        perc_siibra=0.5,
    )

    train_log = nnet.training.train_on_fold(
        model=model,
        fold=fold_data,
        n_epochs=n_epochs,
        end_after=end_after,
        save_name=save_name,
        use_random=use_random,
        use_gaussian=use_gaussian,
        use_synth=use_synth,
    )

    train_log.save(f"{outputs_path}/train_log_{fold}.json")

    experiments.predict_from_model(
        fold=fold,
        fold_data=fold_data,
        outputs_path=outputs_path,
        model=model,
    )
