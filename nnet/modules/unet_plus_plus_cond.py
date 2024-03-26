import torch

import nnet.modules.conditional


class ConditionalUNetPlusPlus(torch.nn.Module):
    def __init__(
        self,
        input_channels: int,
        base_channels: int,
        depth: int,
        num_classes: int,
        embedding_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.conditional_embed = torch.nn.Sequential(
            nnet.modules.conditional.PositionalEmbed3d(embedding_dim),
            torch.nn.Linear(embedding_dim, embedding_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(embedding_dim, embedding_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(embedding_dim, embedding_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(embedding_dim, embedding_dim),
            torch.nn.LeakyReLU(),
        )

        self.input_conv = UNetConvBlock(
            in_channels=input_channels, out_channels=base_channels
        )

        self.backbone = UnetPlusPlusBackbone(
            base_channels=base_channels,
            depth=depth,
            dropout=dropout,
            embedding_dim=embedding_dim,
        )

        self.upsample_stripes = torch.nn.ModuleList(
            [
                UnetPlusPlusUpsampleStripe(
                    base_channels=base_channels,
                    depth=depth - i,
                    n_inputs=i + 1,
                    dropout=dropout,
                )
                for i in range(1, depth)
            ]
        )

        self.final_conv = torch.nn.Conv2d(
            in_channels=base_channels,
            out_channels=num_classes,
            kernel_size=1,
        )

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        condition = self.conditional_embed(condition)
        x = self.input_conv(x)
        residuals = self.backbone(x, condition)
        residuals = [[residual] for residual in residuals]

        for upsample_stripe in self.upsample_stripes:
            outputs = upsample_stripe(residuals)
            for i in range(len(outputs)):
                residuals[i].append(outputs[i])

        return self.final_conv(residuals[0][-1])


class UnetPlusPlusBackbone(torch.nn.Module):
    def __init__(
        self, base_channels: int, depth: int, embedding_dim: int, dropout: float = 0.0
    ):
        super().__init__()

        self.film_blocks = torch.nn.ModuleList(
            [
                nnet.modules.conditional.FiLM(
                    channels=base_channels * 2**i, embedding_dim=embedding_dim
                )
                for i in range(depth)
            ]
        )

        self.process_conv_blocks = torch.nn.ModuleList(
            [
                UNetConvBlock(
                    in_channels=base_channels * 2**i,
                    out_channels=base_channels * 2**i,
                    dropout=dropout,
                )
                for i in range(depth)
            ]
        )

        self.compress_conv_blocks = torch.nn.ModuleList(
            [
                UNetConvBlock(
                    in_channels=base_channels * 2**i,
                    out_channels=base_channels * 2 ** (i + 1),
                    stride=2,
                    kernel_size=4,
                    padding=1,
                    dropout=dropout,
                )
                for i in range(depth - 1)
            ]
        )

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> list[torch.Tensor]:
        features = []

        for i, (film_block, process_block, compress_block) in enumerate(
            zip(self.film_blocks, self.process_conv_blocks, self.compress_conv_blocks)
        ):
            x = film_block(x, condition)
            x = process_block(x)
            features.append(x)
            x = compress_block(x)

        x = self.process_conv_blocks[-1](x)
        features.append(x)

        return features


class UnetPlusPlusUpsampler(torch.nn.Module):
    def __init__(self, base_channels: int, n_inputs: int, dropout: float = 0.0):
        super().__init__()

        self.upsample = torch.nn.ConvTranspose2d(
            in_channels=base_channels * 2,
            out_channels=base_channels,
            kernel_size=2,
            stride=2,
        )

        self.conv_block = UNetConvBlock(
            in_channels=base_channels * n_inputs,
            out_channels=base_channels,
            dropout=dropout,
        )

    def forward(
        self,
        residuals: list[torch.Tensor],
        to_upsample: torch.Tensor,
    ) -> torch.Tensor:
        x = self.upsample(to_upsample)
        x = torch.cat([x] + residuals, dim=1)
        return self.conv_block(x)


class UnetPlusPlusUpsampleStripe(torch.nn.Module):
    def __init__(
        self, base_channels: int, depth: int, n_inputs: int, dropout: float = 0.0
    ):
        super().__init__()

        self.upsamplers = torch.nn.ModuleList(
            [
                UnetPlusPlusUpsampler(
                    base_channels * 2**i, n_inputs=n_inputs, dropout=dropout
                )
                for i in range(depth)
            ]
        )

    def forward(self, residuals: list[list[torch.Tensor]]) -> list[torch.Tensor]:
        outputs = []

        for i in range(len(self.upsamplers)):
            to_upsample = residuals[i + 1][-1]
            cur_residuals = residuals[i]

            output = self.upsamplers[i](cur_residuals, to_upsample)
            outputs.append(output)

        return outputs


class UNetConvBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.conv = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.relu = torch.nn.LeakyReLU()
        self.dropout = torch.nn.Dropout2d(p=dropout)
        self.instance_norm = torch.nn.InstanceNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.relu(self.instance_norm(self.conv(x))))
