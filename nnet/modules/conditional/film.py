import torch
from torch.nn.modules.module import T


class FiLM(torch.nn.Module):
    """FiLM module."""

    def __init__(self, channels: int, embedding_dim: int):
        super().__init__()

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, embedding_dim),
            torch.nn.LeakyReLU(),
            # torch.nn.Linear(embedding_dim, embedding_dim),
            # torch.nn.LeakyReLU(),
            # torch.nn.Linear(embedding_dim, embedding_dim),
            # torch.nn.LeakyReLU(),
            # torch.nn.Linear(embedding_dim, embedding_dim),
            # torch.nn.LeakyReLU(),
            torch.nn.Linear(embedding_dim, channels * 2),
        )
        self._train = True

    def train(self: T, mode: bool = True) -> T:
        """Set the module in training mode."""
        super().train(mode)
        self._train = mode
        return self

    def eval(self: T) -> T:
        """Set the module in evaluation mode."""
        return self.train(False)

    def forward(self, image: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.fc(embedding).chunk(2, dim=1)

        gamma_ranges = gamma.max(dim=0).values - gamma.min(dim=0).values
        beta_ranges = beta.max(dim=0).values - beta.min(dim=0).values

        # if not self._train:
        #    print(f"Gamma: {gamma_ranges.mean()}, Beta: {beta_ranges.mean()}")
        # print("Image shape: ", image.shape)
        # print("Gamma shape: ", gamma[:, :, None, None].shape)
        # print("Beta shape: ", beta[:, :, None, None].shape)

        return image * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]
