import itertools

import torch


class NNUnet(torch.nn.Module):
    def __init__(
        self,
        encoder_map: list[list[int]],
        decoder_map: list[list[int]],
        final_image_size: tuple[int, int],
        hidden_size: int,
        num_classes: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.encoder_layers = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    *[
                        NNUNetConvBlock(
                            in_channels=in_channels, out_channels=out_channels
                        )
                        for in_channels, out_channels in itertools.pairwise(layer)
                    ]
                )
                for layer in encoder_map
            ]
        )

        self.decoder_layers = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    *[
                        NNUNetConvBlock(
                            in_channels=in_channels, out_channels=out_channels
                        )
                        for in_channels, out_channels in itertools.pairwise(layer)
                    ]
                )
                for layer in decoder_map
            ]
        )

        self.downsamples = torch.nn.ModuleList(
            [torch.nn.MaxPool2d(kernel_size=2) for _ in encoder_map[:-1]]
        )

        self.upsamples = torch.nn.ModuleList(
            [
                torch.nn.ConvTranspose2d(
                    in_channels=cur_layer[-1],
                    out_channels=next_layer[-1],
                    kernel_size=2,
                    stride=2,
                )
                for next_layer, cur_layer in itertools.pairwise(decoder_map)
            ]
        )

        self.bottom = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(
                in_features=final_image_size[0]
                * final_image_size[1]
                * encoder_map[-1][-1],
                out_features=hidden_size,
            ),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(
                in_features=hidden_size,
                out_features=final_image_size[0]
                * final_image_size[1]
                * decoder_map[-1][0],
            ),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout),
            torch.nn.Unflatten(
                dim=1,
                unflattened_size=(
                    decoder_map[-1][0],
                    final_image_size[0],
                    final_image_size[1],
                ),
            ),
        )

        self.out_conv = torch.nn.Conv2d(
            in_channels=decoder_map[0][-1], out_channels=num_classes, kernel_size=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoder_outputs = []
        for layer, downsample in zip(self.encoder_layers[:-1], self.downsamples):
            x = layer(x)
            encoder_outputs.append(x)
            x = downsample(x)

        x = self.encoder_layers[-1](x)
        x = self.bottom(x)
        x = self.decoder_layers[-1](x)

        for layer, upsample, encoder_output in zip(
            reversed(self.decoder_layers[:-1]),
            reversed(self.upsamples),
            reversed(encoder_outputs),
        ):
            x = upsample(x)
            x = torch.cat((x, encoder_output), dim=1)
            x = layer(x)

        return self.out_conv(x)


class NNUNetConvBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.conv = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.inst_norm = torch.nn.InstanceNorm2d(num_features=out_channels)
        self.relu = torch.nn.LeakyReLU(1e-2)
        self.dropout = torch.nn.Dropout2d(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.relu(self.inst_norm(self.conv(x))))
    