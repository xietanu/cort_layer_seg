import itertools
from dataclasses import dataclass

import torch
import datasets

import nnet.modules.components


@dataclass
class SegModuleOutput:
    logits: torch.Tensor
    depth_maps: torch.Tensor | None = None
    autoencoded_imgs: torch.Tensor | None = None


class SemantSegUNet(torch.nn.Module):
    def __init__(
        self,
        input_channels: int,
        encoder_map: list[list[int]],
        decoder_map: list[list[int]],
        num_classes: int,
        input_image_size: tuple[int, int] = None,
        dropout: float = 0.0,
        uses_condition: bool = False,
        uses_position: bool = False,
        uses_depth: bool = False,
        embed_dim: int = -1,
        hidden_embed_dim: int = -1,
        use_linear_bridge: bool = False,
        autoencode: bool = False,
    ):
        super().__init__()

        self.uses_condition = uses_condition
        self.uses_position = uses_position
        self.uses_depth = uses_depth
        self.input_img_size = input_image_size
        self.use_linear_bridge = use_linear_bridge
        self.autoencode = autoencode

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
            in_channels=input_channels, out_channels=encoder_map[0][0]
        )

        self.encoder_layers = torch.nn.ModuleList(
            [
                torch.nn.ModuleList(
                    [
                        (
                            nnet.modules.components.ConvBlock(
                                in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=3,
                                padding="same",
                            )
                            if not uses_condition and not uses_position
                            else nnet.modules.components.CondConvBlock(
                                in_channels=in_channels,
                                out_channels=out_channels,
                                embedding_dim=hidden_embed_dim,
                                kernel_size=3,
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

        if self.use_linear_bridge:
            if input_image_size is None:
                raise ValueError(
                    "Input image size must be specified when using a linear bridge."
                )

            downscale_factor = 2 ** (len(encoder_map) - 1)

            if (
                input_image_size[0] % downscale_factor != 0
                or input_image_size[1] % downscale_factor != 0
            ):
                raise ValueError(
                    f"Input image size ({input_image_size}) must be divisible by the downscale factor ({downscale_factor})"
                )

            bottle_img_size = (
                input_image_size[0] // downscale_factor,
                input_image_size[1] // downscale_factor,
            )

            encoder_end_channels = encoder_map[-1][-1]
            decoder_start_channels = decoder_map[-1][0]
            encoder_end_size = (
                encoder_end_channels * bottle_img_size[0] * bottle_img_size[1]
            )
            decoder_start_size = (
                decoder_start_channels * bottle_img_size[0] * bottle_img_size[1]
            )

            self.linear_bridge = torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(
                    encoder_end_size,
                    decoder_start_channels,
                ),
                torch.nn.ReLU(),
                torch.nn.Linear(decoder_start_channels, decoder_start_channels),
                torch.nn.ReLU(),
                torch.nn.Linear(
                    decoder_start_channels,
                    decoder_start_size,
                ),
                torch.nn.ReLU(),
                torch.nn.Unflatten(
                    1,
                    (
                        decoder_start_channels,
                        bottle_img_size[0],
                        bottle_img_size[1],
                    ),
                ),
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
            self.depth_exponents = torch.nn.Parameter(
                torch.ones(num_classes, requires_grad=True) / 100
            )
        else:
            self.head = nnet.modules.components.SegmentationHead(
                in_channels=decoder_map[0][-1], n_classes=num_classes
            )

        if autoencode:
            self.auto_decoder = nnet.modules.components.AutoDecoder(
                decoder_map=decoder_map,
                dropout=dropout,
            )

    def forward(
        self,
        data: datasets.datatypes.SegInputs,
        autoencode_only: bool = False,
    ) -> SegModuleOutput:
        x = self.input_conv(data.input_images.float())

        cond = None
        if self.uses_condition:
            condition = self.cond_embed(data.area_probabilities.float())
            cond = condition
        if self.uses_position:
            position = self.pos_embed(data.position.float())
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

        if self.use_linear_bridge:
            x = self.linear_bridge(x)

        if self.autoencode:
            autoencoded_imgs = self.auto_decoder(x)
            if autoencode_only:
                return SegModuleOutput(
                    logits=None,
                    depth_maps=None,
                    autoencoded_imgs=autoencoded_imgs,
                )
        else:
            autoencoded_imgs = None

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
            logits, depthmap = heads_output
        else:
            logits = heads_output
            depthmap = None

        return SegModuleOutput(
            logits=logits,
            depth_maps=depthmap,
            autoencoded_imgs=autoencoded_imgs,
        )
