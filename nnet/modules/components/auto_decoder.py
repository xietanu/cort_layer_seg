import itertools

import torch

import nnet.modules.components


class AutoDecoder(torch.nn.Module):
    def __init__(
        self,
        decoder_map: list[list[int]],
        dropout: float = 0.0,
    ):
        super().__init__()

        decoder_map = list(reversed(decoder_map))

        self.decoder_layers = torch.nn.Sequential(
            *[
                torch.nn.Sequential(
                    *[
                        nnet.modules.components.ConvBlock(
                            in_channels=in_channels, out_channels=out_channels
                        )
                        for in_channels, out_channels in itertools.pairwise(cur_layer)
                    ],
                    torch.nn.ConvTranspose2d(
                        in_channels=cur_layer[-1],
                        out_channels=next_layer[0],
                        kernel_size=2,
                        stride=2,
                    ),
                )
                for cur_layer, next_layer in itertools.pairwise(decoder_map)
            ],
        )

        self.last_decoder_layer = torch.nn.Sequential(
            *[
                nnet.modules.components.ConvBlock(
                    in_channels=in_channels, out_channels=out_channels
                )
                for in_channels, out_channels in itertools.pairwise(decoder_map[-1])
            ]
        )

        end_channels = decoder_map[-1][-1]

        self.output_conv = torch.nn.Conv2d(
            in_channels=end_channels, out_channels=1, kernel_size=1
        )
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.decoder_layers(x)
        x = self.last_decoder_layer(x)
        x = self.output_conv(x)
        x = self.sigmoid(x)
        return x
