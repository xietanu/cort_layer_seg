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

    def conv_params(self):
        return list(self.conv1.conv_params()) + list(self.conv2.conv_params())

    def film_params(self):
        return list(self.conv1.film_params()) + list(self.conv2.film_params())
