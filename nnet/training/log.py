from __future__ import annotations
import json
from dataclasses import dataclass

import numpy as np
from matplotlib import pyplot as plt

import nnet.loss


class Log:
    train_losses: dict[int, dict[str, float]]
    val_losses: dict[int, dict[str, float]]
    train_accs: dict[int, float]
    val_accs: dict[int, float]

    def __init__(
        self,
        train_losses: dict[int, dict[str, float]] | dict[int, nnet.loss.CombinedLoss],
        val_losses: dict[int, dict[str, float]] | dict[int, nnet.loss.CombinedLoss],
        train_accs: dict[int, float],
        val_accs: dict[int, float],
    ) -> None:
        train_losses = {
            epoch: loss.to_dict() if isinstance(loss, nnet.loss.CombinedLoss) else loss
            for epoch, loss in train_losses.items()
        }
        val_losses = {
            epoch: loss.to_dict() if isinstance(loss, nnet.loss.CombinedLoss) else loss
            for epoch, loss in val_losses.items()
        }

        self.train_losses = train_losses
        self.val_losses = val_losses
        self.train_accs = train_accs
        self.val_accs = val_accs

    def save(self, filepath: str) -> None:
        json.dump(
            {
                "train_losses": self.train_losses,
                "val_losses": self.val_losses,
                "train_accs": self.train_accs,
                "val_accs": self.val_accs,
            },
            open(filepath, "w", encoding="utf-8"),
        )

    @classmethod
    def load(cls, filepath: str) -> Log:
        data = json.load(open(filepath, "r", encoding="utf-8"))
        return cls(
            data["train_losses"],
            data["val_losses"],
            data["train_accs"],
            data["val_accs"],
        )

    def plot(self, smoothing: int = 50, show=True):
        fig, ax = plt.subplots(1, 2, figsize=(18, 8))

        train_losses = {
            k: [] for k in self.train_losses[next(iter(self.train_losses))].keys()
        }

        for epoch, loss in self.train_losses.items():
            for k, v in loss.items():
                train_losses[k].append(v)

        train_losses = {
            k: np.convolve(v, np.ones(smoothing) / smoothing, mode="valid")
            for k, v in train_losses.items()
        }

        train_steps = [int(key) for key in self.train_losses.keys()]
        train_avg_steps = np.convolve(
            train_steps, np.ones(smoothing) / smoothing, mode="valid"
        )
        val_steps = [int(key) for key in self.val_losses.keys()]
        val_losses = {
            k: [] for k in self.val_losses[next(iter(self.val_losses))].keys()
        }

        for epoch, loss in self.val_losses.items():
            for k, v in loss.items():
                val_losses[k].append(v)

        ax[0].set_title("Loss")
        for k, v in train_losses.items():
            ax[0].plot(train_avg_steps, v, label=f"Train - {k}")

        if "total" in val_losses:
            ax[0].plot(
                val_steps, val_losses["total"], label="Val - total", linestyle="--"
            )
        else:
            ax[0].plot(
                val_steps,
                val_losses[next(iter(val_losses))],
                label="Val - total",
                linestyle="--",
            )

        ax[0].legend()

        ax[1].set_title("Accuracy")
        train_avg_acc = np.convolve(
            list(self.train_accs.values()), np.ones(smoothing) / smoothing, mode="valid"
        )
        train_steps = [int(key) for key in self.train_accs.keys()]
        train_avg_step = np.convolve(
            train_steps, np.ones(smoothing) / smoothing, mode="valid"
        )
        val_steps = [int(key) for key in self.val_accs.keys()]

        val_accs = list(self.val_accs.values())

        ax[1].plot(train_avg_step, train_avg_acc, label="Train")
        ax[1].plot(val_steps, val_accs, label="Val", linestyle="--")
        ax[1].legend()

        fig.tight_layout()

        if show:
            plt.show()

        return fig, ax
