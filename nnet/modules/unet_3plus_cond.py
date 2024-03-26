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
        embedding_dim: int,
        hidden_embed_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.conditional_embed = torch.nn.Sequential(
            # nnet.modules.conditional.PositionalEmbed3d(embedding_dim),
            torch.nn.Linear(embedding_dim, hidden_embed_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_embed_dim, hidden_embed_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_embed_dim, hidden_embed_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_embed_dim, hidden_embed_dim),
            torch.nn.LeakyReLU(),
        )

        self.input_conv = nnet.modules.components.ConvBlock(
            in_channels=input_channels, out_channels=base_channels
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

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        cond_embed = self.conditional_embed(condition)
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

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> list[torch.Tensor]:
        features = []

        for i, process_block in enumerate(self.process_conv_blocks[:-1]):
            x = process_block(x, condition)
            features.append(x)
            x = self.downsample(x)

        x = self.process_conv_blocks[-1](x, condition)
        features.append(x)

        return features
