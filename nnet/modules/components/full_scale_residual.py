import torch


class Unet3PlusFullScaleResidual(torch.nn.Module):
    def __init__(
        self, source_depth: int, target_depth: int, total_depth: int, base_channels: int
    ):
        super().__init__()

        if source_depth == target_depth:
            self.resize = torch.nn.Identity()
            input_channels = base_channels * 2**source_depth
        elif source_depth < target_depth:
            self.resize = torch.nn.MaxPool2d(
                kernel_size=2 ** (target_depth - source_depth)
            )
            input_channels = base_channels * 2**source_depth
        else:
            self.resize = torch.nn.UpsamplingBilinear2d(
                scale_factor=2 ** (source_depth - target_depth)
            )
            if source_depth == total_depth - 1:
                input_channels = base_channels * 2**source_depth
            else:
                input_channels = base_channels * total_depth

        self.conv = torch.nn.Conv2d(
            in_channels=input_channels,
            out_channels=base_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.resize(x)
        return self.conv(x)
