import torch

import nnet.modules.components


class SEModule(torch.nn.Module):
    def __init__(self, in_channels: int, reduction: int = 8):
        super().__init__()

        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(in_channels, in_channels // reduction),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(in_channels // reduction, in_channels),
            torch.nn.Sigmoid(),
        )

        self.conv1 = nnet.modules.components.DualConvBlock(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            padding=1,
        )

        self.conv2 = nnet.modules.components.ConvBlock(
            in_channels=in_channels * 2,
            out_channels=in_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        att_path = self.avg_pool(x).view(x.size(0), -1)
        att_path = self.fc(att_path).view(x.size(0), x.size(1), 1, 1)
        att_path = x * att_path

        conv_path = self.conv1(x)

        comb = torch.cat([att_path, conv_path], dim=1)

        return self.conv2(comb)
