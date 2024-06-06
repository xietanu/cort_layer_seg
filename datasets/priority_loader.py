import numpy as np
import torch

import datasets
import cort


class PriorityLoader:
    def __init__(self, dataset: datasets.PatchDataset, batch_size: int):
        self.dataset = dataset
        self.batch_size = batch_size
        self.weights = np.ones(len(dataset))
        self.last_batch = None

    def get_batch(self):
        # weights = self.weights / np.sum(self.weights)
        weights = np.exp(self.weights) / np.sum(np.exp(self.weights))
        self.last_batch = np.random.choice(
            len(self.dataset), size=self.batch_size, p=weights
        )
        batch = tensorify_batch([self.dataset[i] for i in self.last_batch])
        return batch

    def update_weights(self, f1_score):
        self.weights[self.last_batch] = (
            self.weights[self.last_batch] + 1 - f1_score
        ) / 2

    def __len__(self):
        return np.ceil(len(self.dataset) / self.batch_size).astype(int)


def tensorify_batch(batch):
    if isinstance(batch[0][0], torch.Tensor):
        return (
            torch.stack([b[0] for b in batch]),
            (
                torch.stack([b[1][0] for b in batch]),
                torch.stack([b[1][1] for b in batch]),
            ),
        )
    return (
        torch.stack([b[0][0] for b in batch]),
        torch.stack([b[0][1] for b in batch]),
        torch.stack([b[0][2] for b in batch]),
    ), (
        torch.stack([b[1][0] for b in batch]),
        torch.stack([b[1][1] for b in batch]),
    )
