from dataclasses import dataclass

import numpy as np
import torch.optim.lr_scheduler
from tqdm import tqdm

import nnet.protocols
import datasets

TEMP_FILEPATH = "models/unet_temp.pth"


@dataclass
class TrainLog:
    train_losses: dict[int, float]
    val_losses: dict[int, float]
    train_accs: dict[int, float]
    val_accs: dict[int, float]


def train_model(
    model: nnet.protocols.ModelProtocol,
    train_dataloader,
    val_dataloader,
    n_epochs: int = 100,
    save_name: str | None = None,
):
    train_losses = {}
    val_losses = {}
    train_accs = {}
    val_accs = {}

    n_steps = len(train_dataloader) * n_epochs + 1

    step_iter = tqdm(range(n_steps))

    val_loss = 0
    val_acc = 0
    val_acc_err = 0

    train_iter = iter(train_dataloader)

    best_val_loss = np.inf

    for step in step_iter:
        try:
            _, batch_input, batch_seg = next(train_iter)

        except StopIteration:
            train_iter = iter(train_dataloader)
            _, batch_input, batch_seg = next(train_iter)

            all_val_loss = []
            all_val_acc = []
            for _, val_input, val_target in val_dataloader:
                inputs = (
                    datasets.datatypes.DataInputs(*val_input)
                    if isinstance(val_input, (tuple, list))
                    else val_input
                )
                ground_truths = datasets.datatypes.GroundTruths(*val_target)

                cur_val_loss, cur_val_acc, val_acc_err = model.validate(
                    inputs, ground_truths
                )

                all_val_loss.append(cur_val_loss)
                all_val_acc.append(cur_val_acc)
            val_loss = np.mean(all_val_loss)
            val_acc = np.mean(all_val_acc)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model.save(TEMP_FILEPATH)

            val_losses[step] = val_loss
            val_accs[step] = val_acc

        batch_inputs = datasets.datatypes.DataInputs(*batch_input)
        batch_gts = datasets.datatypes.GroundTruths(*batch_seg)

        loss, acc, acc_err = model.train_one_step(batch_inputs, batch_gts)

        lr = model.optimizer.param_groups[0]["lr"]

        step_iter.set_description(
            f"LR: {lr:.6f} | Loss: {loss:.4f}, Acc: {acc:.1%} Acc Err {acc_err:.1%}| Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.1%} Acc Err {val_acc_err:.1%}"
        )

        train_losses[step] = loss
        train_accs[step] = acc

    all_val_loss = []
    all_val_acc = []
    for _, val_input, val_target in val_dataloader:
        inputs = (
            datasets.datatypes.DataInputs(*val_input)
            if isinstance(val_input, (tuple, list))
            else val_input
        )
        ground_truths = datasets.datatypes.GroundTruths(*val_target)
        cur_val_loss, cur_val_acc, val_acc_err = model.validate(inputs, ground_truths)
        all_val_loss.append(cur_val_loss)
        all_val_acc.append(cur_val_acc)
    val_loss = np.mean(all_val_loss)

    if val_loss < best_val_loss:
        model.save(TEMP_FILEPATH)
    else:
        model.load(TEMP_FILEPATH)

    if save_name is not None:
        model.save(f"models/{save_name}.pth")

    return TrainLog(train_losses, val_losses, train_accs, val_accs)
