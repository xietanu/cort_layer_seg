import torch

import nnet.modules.components
import nnet.modules.conditional


class DepthUNet3Plus(torch.nn.Module):
    def __init__(
        self,
        input_channels: int,
        base_channels: int,
        depth: int,
        num_classes: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.input_conv = nnet.modules.components.ConvBlock(
            in_channels=input_channels, out_channels=base_channels
        )

        self.backbone = Unet3PlusBackbone(
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

        self.seg_head = torch.nn.Sequential(
            # nnet.modules.components.DualConvBlock(
            #    in_channels=base_channels * depth,
            #    out_channels=base_channels * depth,
            #    dropout=dropout,
            # ),
            torch.nn.Conv2d(
                in_channels=base_channels * depth,
                out_channels=num_classes,
                kernel_size=1,
            ),
        )

        self.depth_map_head = torch.nn.Sequential(
            # nnet.modules.components.DualConvBlock(
            #    in_channels=base_channels * depth,
            #    out_channels=base_channels * depth,
            #    dropout=dropout,
            # ),
            torch.nn.Conv2d(
                in_channels=base_channels * depth,
                out_channels=num_classes - 1,
                kernel_size=1,
            ),
        )

    def conv_params(self):
        return self.parameters()

    def film_params(self):
        return []

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.input_conv(x)
        residuals = self.backbone(x)
        decoder_outputs = [residuals[-1]]

        for i, decoder in reversed(list(enumerate(self.decoders))):
            output = decoder(residuals[: i + 1] + list(reversed(decoder_outputs)))
            decoder_outputs.append(output)

        segmentation = self.seg_head(decoder_outputs[-1])

        depth_map = self.depth_map_head(decoder_outputs[-1])

        return (
            segmentation,
            depth_map,
        )


class Unet3PlusBackbone(torch.nn.Module):
    def __init__(self, base_channels: int, depth: int, dropout: float = 0.0):
        super().__init__()

        self.process_conv_blocks = torch.nn.ModuleList(
            [
                nnet.modules.components.DualConvBlock(
                    in_channels=max(base_channels * 2 ** (i - 1), base_channels),
                    out_channels=base_channels * 2**i,
                    dropout=dropout,
                )
                for i in range(depth)
            ]
        )

        self.downsample = torch.nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        features = []

        for i, process_block in enumerate(self.process_conv_blocks[:-1]):
            x = process_block(x)
            features.append(x)
            x = self.downsample(x)

        x = self.process_conv_blocks[-1](x)
        features.append(x)

        return features
