import torch

import nnet.modules.conditional


class CondConvBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        embedding_dim: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.conv = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.relu = torch.nn.LeakyReLU()
        self.dropout = torch.nn.Dropout2d(p=dropout)
        self.batch_norm = torch.nn.BatchNorm2d(out_channels)
        self.film = nnet.modules.conditional.FiLM(out_channels, embedding_dim)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.film(x, condition)
        x = self.relu(x)
        x = self.dropout(x)
        return x

    def conv_params(self):
        return list(self.conv.parameters()) + list(self.batch_norm.parameters())

    def film_params(self):
        return list(self.film.parameters())
