import torch


class FiLM(torch.nn.Module):
    """FiLM module."""

    def __init__(self, channels: int, embedding_dim: int):
        super().__init__()

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, channels * 2),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(channels * 2, channels * 2),
        )

    def forward(self, image: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.fc(embedding).chunk(2, dim=1)

        # print("Image shape: ", image.shape)
        # print("Gamma shape: ", gamma[:, :, None, None].shape)
        # print("Beta shape: ", beta[:, :, None, None].shape)

        return image * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]
