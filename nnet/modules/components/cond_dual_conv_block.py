import torch

import nnet.modules.components
import nnet.modules.conditional


class CondDualConvBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        embedding_dim: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dropout: float = 0.0,
        dilation: int = 1,
    ):
        super().__init__()

        self.conv1 = nnet.modules.components.CondConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dropout=dropout,
            dilation=dilation,
            embedding_dim=embedding_dim,
        )

        self.film = nnet.modules.conditional.FiLM(out_channels, embedding_dim)

        self.conv2 = nnet.modules.components.CondConvBlock(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dropout=dropout,
            dilation=dilation,
            embedding_dim=embedding_dim,
        )

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, condition)
        x = self.conv2(x, condition)
        return x
