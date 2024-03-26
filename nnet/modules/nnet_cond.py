import torch

import nnet.modules.components
import nnet.modules.conditional


class ConditionalNNet(torch.nn.Module):
    def __init__(
        self,
        input_channels: int,
        base_channels: int,
        depth: int,
        num_classes: int,
        embed_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.conditional_embed = torch.nn.Sequential(
            nnet.modules.conditional.PositionalEmbed3d(embed_dim),
            torch.nn.Linear(embed_dim, embed_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(embed_dim, embed_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(embed_dim, embed_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(embed_dim, embed_dim),
            torch.nn.LeakyReLU(),
        )

        self.backbone = nnet.modules.components.CondNNetDualEncoder(
            in_channels=input_channels,
            base_channels=base_channels,
            depth=depth,
            dropout=dropout,
            embed_dim=embed_dim,
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
        residuals = self.backbone(x, cond_embed)
        decoder_outputs = [residuals[-1]]

        for i, decoder in reversed(list(enumerate(self.decoders))):
            output = decoder(residuals[: i + 1] + list(reversed(decoder_outputs)))
            decoder_outputs.append(output)

        return self.final_conv(decoder_outputs[-1])
