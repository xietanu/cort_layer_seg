import numpy as np
from tqdm import tqdm

import nnet.protocols
import nnet.training
import datasets

TEMP_FILEPATH = "models/unet_temp.pth"


def train_on_fold_priority(
    model: nnet.protocols.ModelProtocol,
    fold: datasets.Fold,
    n_epochs: int = 100,
    save_name: str | None = None,
):
    """Train a model on a fold_data."""
    train_losses = {}
    val_losses = {}

    dataset_size = len(fold.train_dataloader)

    n_steps = dataset_size * n_epochs + 1

    step_iter = tqdm(range(n_steps))

    val_loss = 0
    val_acc = 0

    best_val_loss = np.inf

    for step in step_iter:
        if (step + 1) % dataset_size == 0:

            all_val_loss = []
            all_val_acc = []
            for val_input, val_target in fold.val_dataloader:
                cur_val_loss, cur_val_acc = model.validate(val_input, val_target)
                all_val_loss.append(cur_val_loss)
                all_val_acc.append(cur_val_acc)
            val_loss = np.mean(all_val_loss)
            val_acc = np.mean(all_val_acc)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model.save(TEMP_FILEPATH)

            val_losses[step] = val_loss

        batch_input, batch_seg = fold.train_dataloader.get_batch()

        loss, acc = model.train_one_step(batch_input, batch_seg)

        fold.train_dataloader.update_weights(acc)

        step_iter.set_description(
            f"Loss: {loss:.4f}, Acc: {acc:.1%} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.1%}"
        )

        train_losses[step] = loss

    all_val_loss = []
    all_val_acc = []
    for val_input, val_target in fold.val_dataloader:
        cur_val_loss, cur_val_acc = model.validate(val_input, val_target)
        all_val_loss.append(cur_val_loss)
        all_val_acc.append(cur_val_acc)
    val_loss = np.mean(all_val_loss)

    if val_loss < best_val_loss:
        model.save(TEMP_FILEPATH)
    else:
        model.load(TEMP_FILEPATH)

    if save_name is not None:
        model.save(f"models/{save_name}.pth")
