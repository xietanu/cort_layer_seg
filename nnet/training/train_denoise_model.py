from dataclasses import dataclass

import numpy as np
from tqdm import tqdm

import nnet.protocols


@dataclass
class TrainLog:
    train_losses: dict[int, float]
    val_losses: dict[int, float]
    train_accs: dict[int, float]
    val_accs: dict[int, float]


def train_denoise_model(
    model: nnet.protocols.DenoiseModelProtocol,
    train_dataloader,
    val_dataloader,
    n_epochs: int,
    save_name: str,
):
    train_losses = {}
    val_losses = {}
    train_accs = {}
    val_accs = {}

    n_steps = len(train_dataloader) * n_epochs + 1

    step_rep = tqdm(range(n_steps))

    val_loss = 0
    val_acc = 0

    best_val_loss = np.inf

    step = 0

    for epoch in range(n_epochs):
        for inputs, seg_gt in train_dataloader:
            if isinstance(inputs, (tuple, list)):
                logits, probs, locations = inputs
            else:
                logits = inputs
                probs = None
                locations = None
            loss, acc = model.train_one_step(logits, seg_gt, probs, locations)
            lr = model.optimizer.param_groups[0]["lr"]
            step_rep.set_description(
                f"LR: {lr:.6f} | Loss: {loss:.4f}, Acc: {acc:.1%}| Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.1%}"
            )
            step_rep.update(1)
            step += 1

            train_losses[step] = loss
            train_accs[step] = acc

        all_val_loss = []
        all_val_acc = []
        for inputs, seg_gt in val_dataloader:
            if isinstance(inputs, (tuple, list)):
                logits, probs, locations = inputs
            else:
                logits = inputs
                probs = None
                locations = None
            cur_val_loss, cur_val_acc = model.validate(logits, seg_gt, probs, locations)

            all_val_loss.append(cur_val_loss)
            all_val_acc.append(cur_val_acc)
        val_loss = np.mean(all_val_loss)
        val_acc = np.mean(all_val_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save(f"models/{save_name}.pth")

        val_losses[step] = val_loss
        val_accs[step] = val_acc

    step_rep.close()

    return TrainLog(train_losses, val_losses, train_accs, val_accs)
