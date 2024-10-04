import argparse
import json

import datasets
import experiments


CONFIGS_PATH = "experiments/configs"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_name", type=str)
    parser.add_argument("fold", type=int)

    args = parser.parse_args()

    config_name = args.config_name
    fold = args.fold

    config = json.load(open(f"{CONFIGS_PATH}/{config_name}.json"))

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
        padded_size=(256, 128),
    )

    experiments.predict_from_model(
        fold, fold_data, outputs_path=f"experiments/results/{config_name}"
    )


if __name__ == "__main__":
    main()
