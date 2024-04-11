import cort
import datasets
import nnet.modules.conditional
import torch

EMBED_DIM = 48


class PosCondition(datasets.protocols.Condition):
    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim

    def __call__(self, cort_patch: cort.CorticalPatch) -> torch.Tensor:
        return torch.tensor(
            [
                cort_patch.x,
                cort_patch.y,
                cort_patch.z,
            ],
            dtype=torch.float64,
        )

    def get_embed_network(self):
        return torch.nn.Sequential(
            # nnet.modules.conditional.PositionalEmbed3d(EMBED_DIM),
            torch.nn.Linear(3, self.embedding_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.embedding_dim, self.embedding_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.embedding_dim, self.embedding_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.embedding_dim, self.embedding_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.embedding_dim, self.embedding_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.embedding_dim, self.embedding_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.embedding_dim, self.embedding_dim),
        )
