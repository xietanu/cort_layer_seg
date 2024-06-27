import json
import os

import numpy as np
import torch

import experiments
import nnet.training
import datasets

CONFIGS_PATH = "experiments/configs"


def train_denoise_model(
    config_name: str,
    n_epochs: int,
    save_name: str,
    continue_training: bool = False,
):
    """Run an experiment on a fold_data."""
    config = json.load(open(f"{CONFIGS_PATH}/{config_name}.json"))

    model = experiments.create_denoise_model_from_config(config)

    if continue_training:
        model.load(f"models/{save_name}.pth")

    train_dataset = datasets.DenoiseDataset(
        "data/denoise_data", noise_level=1.75, use_conditions=config["conditional"]
    )
    val_dataset = datasets.DenoiseDataset(
        "data/denoise_data_val", noise_level=0.0, use_conditions=config["conditional"]
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=4, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=False)

    train_log = nnet.training.train_denoise_model(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        n_epochs=n_epochs,
        save_name=save_name,
    )

    outputs_path = f"experiments/results/{save_name}"
    os.makedirs(outputs_path, exist_ok=True)

    print("Saving training logs...")
    os.makedirs((os.path.join(outputs_path, "logs")), exist_ok=True)
    json.dump(
        train_log.train_losses,
        open(os.path.join(outputs_path, "logs", "train_losses.json"), "w"),
    )
    json.dump(
        train_log.val_losses,
        open(os.path.join(outputs_path, "logs", "val_losses.json"), "w"),
    )
    json.dump(
        train_log.train_accs,
        open(os.path.join(outputs_path, "logs", "train_accs.json"), "w"),
    )
    json.dump(
        train_log.val_accs,
        open(os.path.join(outputs_path, "logs", "val_accs.json"), "w"),
    )

    print("Done!")
