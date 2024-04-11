import torch


class PositionalEmbed3d(torch.nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()

        if embedding_dim % 6 != 0:
            raise ValueError("Embedding dimension must be divisible by 6.")

        self.embedding_dim = embedding_dim

    def forward(self, coords: torch.Tensor):
        embedding = torch.zeros(
            (coords.shape[0], self.embedding_dim), device=coords.device
        )

        for dim in range(0, self.embedding_dim // 3, 2):
            embedding[:, dim] = torch.sin(
                coords[:, 0] / 100 * (dim / self.embedding_dim) * torch.pi
            )
            embedding[:, dim + 1] = torch.cos(
                coords[:, 0] / 100 * (dim / self.embedding_dim) * torch.pi
            )
            embedding[:, dim + self.embedding_dim // 3] = torch.sin(
                coords[:, 1] / 100 * (dim / self.embedding_dim) * torch.pi
            )
            embedding[:, dim + 1 + self.embedding_dim // 3] = torch.cos(
                coords[:, 1] / 100 * (dim / self.embedding_dim) * torch.pi
            )
            embedding[:, dim + 2 * self.embedding_dim // 3] = torch.sin(
                coords[:, 2] / 100 * (dim / self.embedding_dim) * torch.pi
            )
            embedding[:, dim + 1 + 2 * self.embedding_dim // 3] = torch.cos(
                coords[:, 2] / 100 * (dim / self.embedding_dim) * torch.pi
            )

        return embedding
