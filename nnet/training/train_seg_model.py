import itertools
from dataclasses import dataclass

import numpy as np
import torch.optim.lr_scheduler
from tqdm import tqdm
import nnet.loss

import nnet.protocols
import datasets

TEMP_FILEPATH = "models/unet_temp.pth"


def train_seg_model(
    model: nnet.protocols.SegModelProtocol,
    train_dataloader,
    val_dataloader,
    random_dataloader=None,
    gaussian_dataloader=None,
    syn_dataloader=None,
    n_epochs: int = -1,
    end_after: int = -1,
    save_name: str | None = None,
):
    if n_epochs <= 0 and end_after <= 0:
        raise ValueError("Must specify either n_epochs or end_after")

    train_losses = {}
    val_losses = {}
    train_accs = {}
    val_accs = {}

    if n_epochs > 0:
        n_steps = len(train_dataloader) * n_epochs + 1

        step_iter = tqdm(range(n_steps))
    else:
        step_iter = tqdm(itertools.count())

    val_loss = nnet.loss.CombinedLoss()
    val_acc = 0
    val_acc_err = 0
    not_improved = 0

    train_iter = iter(train_dataloader)

    if random_dataloader is not None:
        random_iter = itertools.cycle(random_dataloader)
    else:
        random_iter = None

    if gaussian_dataloader is not None:
        gaussian_iter = itertools.cycle(gaussian_dataloader)
    else:
        gaussian_iter = None

    if syn_dataloader is not None:
        syn_iter = itertools.cycle(syn_dataloader)
    else:
        syn_iter = None

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
                    datasets.datatypes.SegInputs(*val_input)
                    if isinstance(val_input, (tuple, list))
                    else val_input
                )
                ground_truths = datasets.datatypes.SegGroundTruths(*val_target)

                cur_val_loss, cur_val_acc = model.validate(inputs, ground_truths)

                all_val_loss.append(cur_val_loss)
                all_val_acc.append(cur_val_acc)
            val_loss = nnet.loss.CombinedLoss.mean(all_val_loss)
            val_acc = np.mean(all_val_acc)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model.save(TEMP_FILEPATH)
                not_improved = 0
            else:
                not_improved += 1
                if end_after > 0 and not_improved >= end_after:
                    break

            val_losses[step] = val_loss
            val_accs[step] = val_acc

        batch_inputs = datasets.datatypes.SegInputs(*batch_input)
        batch_gts = datasets.datatypes.SegGroundTruths(*batch_seg)

        loss, acc = model.train_one_step(
            batch_inputs, batch_gts, step=step / len(train_dataloader)
        )

        if random_iter is not None:
            for __ in range(
                train_dataloader.batch_size // random_dataloader.batch_size
            ):
                _, random_input, random_seg = next(random_iter)
                random_inputs = datasets.datatypes.SegInputs(*random_input)
                model.train_one_step(random_inputs, None, step=None)

        if gaussian_iter is not None:
            _, gaussian_input, gaussian_seg = next(gaussian_iter)
            gaussian_inputs = datasets.datatypes.SegInputs(*gaussian_input)
            gaussian_gts = datasets.datatypes.SegGroundTruths(*gaussian_seg)
            model.train_one_step(gaussian_inputs, gaussian_gts, step=None)

        if syn_iter is not None:
            _, syn_input, syn_seg = next(syn_iter)
            syn_inputs = datasets.datatypes.SegInputs(*syn_input)
            syn_gts = datasets.datatypes.SegGroundTruths(*syn_seg)
            model.train_one_step(syn_inputs, syn_gts, step=None)

        lr = model.optimizer.param_groups[0]["lr"]

        if n_epochs > 0:
            step_iter.set_description(
                f"LR: {lr:.6f} | Loss: {loss.total:.4f}, Acc: {acc:.1%}| Val Loss: {val_loss.total:.4f}, Val Acc: {val_acc:.1%}"
            )
        else:
            step_iter.set_description(
                f"LR: {lr:.6f} | Loss: {loss.total:.4f}, Acc: {acc:.1%}| Val Loss: {val_loss.total:.4f}, Val Acc: {val_acc:.1%} | Epochs: {step / len(train_dataloader):.2f}, since last improvement: {not_improved}"
            )

        train_losses[step] = loss
        train_accs[step] = acc

    all_val_loss = []
    all_val_acc = []
    for _, val_input, val_target in val_dataloader:
        inputs = (
            datasets.datatypes.SegInputs(*val_input)
            if isinstance(val_input, (tuple, list))
            else val_input
        )
        ground_truths = datasets.datatypes.SegGroundTruths(*val_target)
        cur_val_loss, cur_val_acc = model.validate(inputs, ground_truths)
        all_val_loss.append(cur_val_loss)
        all_val_acc.append(cur_val_acc)
    val_loss = nnet.loss.CombinedLoss.mean(all_val_loss)

    if val_loss < best_val_loss:
        model.save(TEMP_FILEPATH)
    else:
        model.load(TEMP_FILEPATH)

    if save_name is not None:
        model.save(f"models/{save_name}.pth")

    return nnet.training.Log(train_losses, val_losses, train_accs, val_accs)
