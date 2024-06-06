import torch

import nnet.modules.components
import nnet.modules.conditional


class ConditionalUNet3Plus(torch.nn.Module):
    def __init__(
        self,
        input_channels: int,
        base_channels: int,
        depth: int,
        num_classes: int,
        embed_dim: int,
        hidden_embed_dim: int,
        uses_condition: bool = False,
        uses_position: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()

        if not uses_condition and not uses_position:
            raise ValueError(
                "At least one of uses_condition and uses_position must be True"
            )

        self.input_conv = nnet.modules.components.ConvBlock(
            in_channels=input_channels, out_channels=base_channels
        )
        # mid_embed_dim = max(hidden_embed_dim, embed_dim)

        if uses_condition:
            self.cond_embed = torch.nn.Sequential(
                torch.nn.Linear(embed_dim, hidden_embed_dim),
                # torch.nn.ReLU(),
                # torch.nn.Linear(mid_embed_dim, mid_embed_dim),
                # torch.nn.ReLU(),
                # torch.nn.Linear(mid_embed_dim, mid_embed_dim),
                # torch.nn.ReLU(),
                # torch.nn.Linear(mid_embed_dim, hidden_embed_dim),
                # torch.nn.ReLU(),
            )

        if uses_position:
            self.pos_embed = nnet.modules.conditional.PositionalEmbed3d(
                embedding_dim=hidden_embed_dim,
            )

        self.backbone = CondUnet3PlusBackbone(
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

        self.final_conv = torch.nn.Conv2d(
            in_channels=base_channels * depth,
            out_channels=num_classes,
            kernel_size=1,
        )
        self.uses_condition = uses_condition
        self.uses_position = uses_position

    def conv_params(self):
        params = (
            list(self.input_conv.parameters())
            + list(self.final_conv.parameters())
            + list(self.decoders.parameters())
            + list(self.backbone.conv_params())
        )
        return params

    def film_params(self):
        return (
            list(self.backbone.film_params())
            + (list(self.cond_embed.parameters()) if self.uses_condition else [])
            + (list(self.pos_embed.parameters()) if self.uses_position else [])
        )

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor, pos: torch.Tensor
    ) -> torch.Tensor:
        if not self.uses_condition and not self.uses_position:
            raise ValueError(
                "At least one of uses_condition and uses_position must be True"
            )

        if self.uses_condition:
            label_embed = self.cond_embed(cond)
            cond_embed = label_embed
        if self.uses_position:
            pos_embed = self.pos_embed(pos)
            cond_embed = pos_embed

        if self.uses_condition and self.uses_position:
            cond_embed = label_embed + pos_embed

        x = self.input_conv(x)
        residuals = self.backbone(x, cond_embed)
        decoder_outputs = [residuals[-1]]

        for i, decoder in reversed(list(enumerate(self.decoders))):
            output = decoder(residuals[: i + 1] + list(reversed(decoder_outputs)))
            decoder_outputs.append(output)

        return self.final_conv(decoder_outputs[-1])


class CondUnet3PlusBackbone(torch.nn.Module):
    def __init__(
        self, base_channels: int, depth: int, embed_dim: int, dropout: float = 0.0
    ):
        super().__init__()

        self.process_conv_blocks = torch.nn.ModuleList(
            [
                nnet.modules.components.CondDualConvBlock(
                    in_channels=max(base_channels * 2 ** (i - 1), base_channels),
                    out_channels=base_channels * 2**i,
                    dropout=dropout,
                    embedding_dim=embed_dim,
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

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> list[torch.Tensor]:
        features = []

        for i, process_block in enumerate(self.process_conv_blocks[:-1]):
            x = process_block(x, condition)
            features.append(x)
            x = self.downsample(x)

        x = self.process_conv_blocks[-1](x, condition)
        features.append(x)

        return features
