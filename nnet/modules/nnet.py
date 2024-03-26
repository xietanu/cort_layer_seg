import torch

import nnet.modules.components


class NNet(torch.nn.Module):
    def __init__(
        self,
        input_channels: int,
        base_channels: int,
        depth: int,
        num_classes: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.backbone = nnet.modules.components.NNetDualEncoder(
            in_channels=input_channels,
            base_channels=base_channels,
            depth=depth,
            dropout=dropout,
        )

        self.decoders = torch.nn.ModuleList(
            [
                nnet.modules.components.Unet3PlusDecoder(
                    base_channels=base_channels,
                    depth=i,
                    backbone_depth=depth,
                    dropout=dropout,
                )
                for i in range(depth - 1)
            ]
        )

        self.final_conv = torch.nn.Conv2d(
            in_channels=base_channels * depth,
            out_channels=num_classes,
            kernel_size=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residuals = self.backbone(x)
        decoder_outputs = [residuals[-1]]

        for i, decoder in reversed(list(enumerate(self.decoders))):
            output = decoder(residuals[: i + 1] + list(reversed(decoder_outputs)))
            decoder_outputs.append(output)

        return self.final_conv(decoder_outputs[-1])
