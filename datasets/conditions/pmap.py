import cort
import datasets
import nnet.modules.conditional
import torch


class PMapCondition(datasets.protocols.Condition):
    def __init__(self, cond_dim: int, embedding_dim: int):
        self.embedding_dim = embedding_dim
        self.cond_dim = cond_dim

    def __call__(self, cort_patch: cort.CorticalPatch) -> torch.Tensor:
        return torch.tensor(cort_patch.region_probs, dtype=torch.float64)

    def get_embed_network(self):
        return torch.nn.Sequential(
            # nnet.modules.conditional.PositionalEmbed3d(EMBED_DIM),
            torch.nn.Linear(self.cond_dim, self.embedding_dim),
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
