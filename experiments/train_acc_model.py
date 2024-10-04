import json
import os

import numpy as np
import torch

import evaluate
import experiments
import nnet.training
import datasets

CONFIGS_PATH = "experiments/configs"


def train_acc_model(
    config_name: str,
    n_epochs: int,
    save_name: str,
    continue_training: bool = False,
):
    """Run an experiment on a fold_data."""
    config = json.load(open(f"{CONFIGS_PATH}/{config_name}.json"))

    model = experiments.create_acc_model_from_config(config)

    if continue_training:
        model.load(f"models/{save_name}.pth")

    train_dataset = datasets.AccuracyDataset(
        "data/denoise_data", noise_level=1.75, use_conditions=config["conditional"]
    )
    val_dataset = datasets.AccuracyDataset(
        "data/denoise_data_val", noise_level=0, use_conditions=config["conditional"]
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=4, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=False)

    train_log = nnet.training.train_accuracy_model(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        n_epochs=n_epochs,
        save_name=save_name,
    )

    outputs_path = f"experiments/results/{save_name}"
    os.makedirs(outputs_path, exist_ok=True)

    print("Predicting validation set...")
    pred_pairs = []
    for inputs, seg_gt in val_loader:
        if isinstance(inputs, (tuple, list)):
            logits, probs, locations = inputs
        else:
            logits = inputs
            probs = None
            locations = None
        pred_scores = model.predict(logits, probs, locations)
        for i in range(len(seg_gt)):
            pred_seg = logits[i].argmax(0)
            gt_f1 = evaluate.f1_score(
                pred_seg[None, :, :],
                seg_gt[i, None, :, :],
                n_classes=8,
                ignore_index=8,
            )
            pred_pairs.append({"gt": gt_f1, "pred": pred_scores[i].item()})
    print("Saving predictions...")
    json.dump(pred_pairs, open(os.path.join(outputs_path, "val_predictions.json"), "w"))

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
