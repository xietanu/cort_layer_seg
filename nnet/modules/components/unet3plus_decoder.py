import torch

import nnet.modules.components


class Unet3PlusDecoder(torch.nn.Module):
    def __init__(
        self, base_channels: int, depth: int, backbone_depth: int, dropout: float = 0.0
    ):
        super().__init__()

        self.full_scale_residual_blocks = torch.nn.ModuleList(
            [
                nnet.modules.components.Unet3PlusFullScaleResidual(
                    source_depth=i,
                    target_depth=depth,
                    base_channels=base_channels,
                    total_depth=backbone_depth,
                )
                for i in range(backbone_depth)
            ]
        )

        self.conv_block = nnet.modules.components.DualConvBlock(
            in_channels=base_channels * backbone_depth,
            out_channels=base_channels * backbone_depth,
            dropout=dropout,
        )

    def forward(self, residuals: list[torch.Tensor]) -> torch.Tensor:
        full_scale_residuals = [
            full_scale_residual_block(residual)
            for full_scale_residual_block, residual in zip(
                self.full_scale_residual_blocks, residuals
            )
        ]

        x = torch.cat(full_scale_residuals, dim=1)

        return self.conv_block(x)
