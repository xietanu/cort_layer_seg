import itertools

import torch

import nnet.modules.components


class SemantSegUNet(torch.nn.Module):
    def __init__(
        self,
        encoder_map: list[list[int]],
        decoder_map: list[list[int]],
        num_classes: int,
        dropout: float = 0.0,
        uses_condition: bool = False,
        uses_position: bool = False,
        uses_depth: bool = False,
        embed_dim: int = -1,
        hidden_embed_dim: int = -1,
    ):
        super().__init__()

        self.uses_condition = uses_condition
        self.uses_position = uses_position
        self.uses_depth = uses_depth

        if uses_condition:
            if embed_dim == -1:
                raise ValueError("Embedding dimension must be specified.")
            if hidden_embed_dim == -1:
                raise ValueError("Hidden embedding dimension must be specified.")
            mid_embed_dim = max(hidden_embed_dim, embed_dim)
            self.cond_embed = torch.nn.Sequential(
                torch.nn.Linear(embed_dim, mid_embed_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(mid_embed_dim, mid_embed_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(mid_embed_dim, mid_embed_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(mid_embed_dim, hidden_embed_dim),
                torch.nn.ReLU(),
            )

        if uses_position:
            if embed_dim == -1:
                raise ValueError("Embedding dimension must be specified.")
            if hidden_embed_dim == -1:
                raise ValueError("Hidden embedding dimension must be specified.")
            self.pos_embed = torch.nn.Sequential(
                torch.nn.Linear(3, hidden_embed_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_embed_dim, hidden_embed_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_embed_dim, hidden_embed_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_embed_dim, hidden_embed_dim),
                torch.nn.ReLU(),
            )

        self.input_conv = nnet.modules.components.ConvBlock(
            in_channels=1, out_channels=encoder_map[0][0]
        )

        self.encoder_layers = torch.nn.ModuleList(
            [
                torch.nn.ModuleList(
                    [
                        (
                            nnet.modules.components.ConvBlock(
                                in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=3 if i > 0 else 5,
                                padding="same",
                            )
                            if not uses_condition and not uses_position
                            else nnet.modules.components.CondConvBlock(
                                in_channels=in_channels,
                                out_channels=out_channels,
                                embedding_dim=hidden_embed_dim,
                                kernel_size=3 if i > 0 else 5,
                                padding="same",
                            )
                        )
                        for i, (in_channels, out_channels) in enumerate(
                            itertools.pairwise(layer)
                        )
                    ]
                )
                for layer in encoder_map
            ]
        )

        self.decoder_layers = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    *[
                        nnet.modules.components.ConvBlock(
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

        if uses_depth:
            self.head = nnet.modules.components.DualHead(
                in_channels=decoder_map[0][-1], n_classes=num_classes
            )
        else:
            self.head = nnet.modules.components.SegmentationHead(
                in_channels=decoder_map[0][-1], n_classes=num_classes
            )

        self.accuracy_head = nnet.modules.components.BottomAccuracyHead(
            in_channels=encoder_map[-1][-1]
        )

    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor | None = None,
        position: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.input_conv(x)

        cond = None
        if self.uses_condition:
            condition = self.cond_embed(condition)
            cond = condition
        if self.uses_position:
            position = self.pos_embed(position)
            cond = position
        if self.uses_condition and self.uses_position:
            cond = condition + position

        encoder_outputs = []
        for layer, downsample in zip(self.encoder_layers[:-1], self.downsamples):
            if self.uses_condition or self.uses_position:
                for conv in layer:
                    x = conv(x, cond)
            else:
                for conv in layer:
                    x = conv(x)
            encoder_outputs.append(x)
            x = downsample(x)

        if self.uses_condition or self.uses_position:
            for conv in self.encoder_layers[-1]:
                x = conv(x, cond)
        else:
            for conv in self.encoder_layers[-1]:
                x = conv(x)

        accuracy = self.accuracy_head(x)

        x = self.decoder_layers[-1](x)

        for layer, upsample, encoder_output in zip(
            reversed(self.decoder_layers[:-1]),
            reversed(self.upsamples),
            reversed(encoder_outputs),
        ):
            x = upsample(x)
            x = torch.cat((x, encoder_output), dim=1)
            x = layer(x)

        heads_output = self.head(x)

        if self.uses_depth:
            return heads_output[0], accuracy, heads_output[1]
        else:
            return heads_output, accuracy
