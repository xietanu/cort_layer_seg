import torch
from sympy.core.numbers import NegativeOne

import nnet.modules.components
import nnet.modules.conditional


class UNet3Plus(torch.nn.Module):
    def __init__(
        self,
        input_channels: int,
        base_channels: int,
        depth: int,
        num_classes: int,
        embed_dim: int = -1,
        hidden_embed_dim: int = -1,
        uses_condition: bool = False,
        uses_position: bool = False,
        uses_depth: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.input_conv = nnet.modules.components.ConvBlock(
            in_channels=input_channels, out_channels=base_channels
        )

        if uses_condition:
            if embed_dim == -1:
                raise ValueError("Embedding dimension must be specified.")
            if hidden_embed_dim == -1:
                raise ValueError("Hidden embedding dimension must be specified.")
            mid_embed_dim = max(hidden_embed_dim, embed_dim)
            self.cond_embed = torch.nn.Sequential(
                torch.nn.Linear(embed_dim, hidden_embed_dim),
                #    torch.nn.ReLU(),
                #    torch.nn.Linear(mid_embed_dim, mid_embed_dim),
                #    torch.nn.ReLU(),
                #    torch.nn.Linear(mid_embed_dim, mid_embed_dim),
                #    torch.nn.ReLU(),
                #    torch.nn.Linear(mid_embed_dim, hidden_embed_dim),
                #    torch.nn.ReLU(),
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

        self.backbone = Unet3PlusBackbone(
            base_channels=base_channels,
            depth=depth,
            dropout=dropout,
            embed_dim=hidden_embed_dim,
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

        if uses_depth:
            self.head = nnet.modules.components.DualHead(
                in_channels=base_channels * depth, n_classes=num_classes
            )
        else:
            self.head = nnet.modules.components.SegmentationHead(
                in_channels=base_channels * depth, n_classes=num_classes
            )

        self.uses_condition = uses_condition
        self.uses_position = uses_position
        self.uses_depth = uses_depth

    def conv_params(self):
        params = (
            list(self.input_conv.parameters())
            + list(self.head.conv_params())
            + list(self.decoders.parameters())
            + list(self.backbone.conv_params())
        )
        return params

    def film_params(self):
        return (
            list(self.backbone.film_params())
            + (list(self.cond_embed.parameters()) if self.uses_condition else [])
            + (list(self.pos_embed.parameters()) if self.uses_position else [])
            + list(self.head.film_params())
        )

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor | None = None,
        pos: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.input_conv(x)

        cond_embed = None

        if self.uses_condition:
            label_embed = self.cond_embed(cond)
            cond_embed = label_embed
        if self.uses_position:
            pos_embed = self.pos_embed(pos)
            cond_embed = pos_embed

        if self.uses_condition and self.uses_position:
            cond_embed = label_embed + pos_embed

        residuals = self.backbone(x, cond_embed)

        decoder_outputs = [residuals[-1]]

        for i, decoder in reversed(list(enumerate(self.decoders))):
            output = decoder(residuals[: i + 1] + list(reversed(decoder_outputs)))
            decoder_outputs.append(output)

        return self.head(decoder_outputs[-1])


class Unet3PlusBackbone(torch.nn.Module):
    def __init__(
        self, base_channels: int, depth: int, embed_dim: int, dropout: float = 0.0
    ):
        super().__init__()

        self.process_conv_blocks = torch.nn.ModuleList(
            [
                (
                    nnet.modules.components.CondDualConvBlock(
                        in_channels=max(base_channels * 2 ** (i - 1), base_channels),
                        out_channels=base_channels * 2**i,
                        dropout=dropout,
                        embedding_dim=embed_dim,
                    )
                    if embed_dim > 0
                    else nnet.modules.components.DualConvBlock(
                        in_channels=max(base_channels * 2 ** (i - 1), base_channels),
                        out_channels=base_channels * 2**i,
                        dropout=dropout,
                    )
                )
                for i in range(depth)
            ]
        )

        self.downsample = torch.nn.MaxPool2d(kernel_size=2, stride=2)

    def conv_params(self):
        params = list(self.downsample.parameters())
        for process_block in self.process_conv_blocks:
            params += process_block.conv_params()
        return params

    def film_params(self):
        params = []
        for process_block in self.process_conv_blocks:
            params += process_block.film_params()
        return params

    def forward(
        self, x: torch.Tensor, condition: torch.Tensor | None = None
    ) -> list[torch.Tensor]:
        features = []

        for i, process_block in enumerate(self.process_conv_blocks[:-1]):
            inputs = (x, condition) if condition is not None else (x,)
            x = process_block(*inputs)
            features.append(x)
            x = self.downsample(x)

        inputs = (x, condition) if condition is not None else (x,)
        x = self.process_conv_blocks[-1](*inputs)
        features.append(x)

        return features
