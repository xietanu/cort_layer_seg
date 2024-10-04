import torch

import nnet.modules.components


class ResNetBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernal_size: int = 3,
        padding: str = "same",
    ):
        super().__init__()

        self.conv1 = nnet.modules.components.ConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernal_size,
            padding=padding,
        )
        self.conv2 = nnet.modules.components.ConvBlock(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernal_size,
            padding=padding,
        )
        if in_channels != out_channels:
            self.shortcut = nnet.modules.components.ConvBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                padding=padding,
            )
        else:
            self.shortcut = torch.nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x += identity
        return x
