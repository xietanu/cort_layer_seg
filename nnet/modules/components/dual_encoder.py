import torch

import nnet.modules.components


class NNetDualEncoder(torch.nn.Module):
    def __init__(
        self, in_channels: int, base_channels: int, depth: int, dropout: float = 0.0
    ):
        super().__init__()

        self.enc_blocks = torch.nn.ModuleList(
            [
                nnet.modules.components.DualConvBlock(
                    in_channels=in_channels,
                    out_channels=base_channels,
                    dropout=dropout,
                )
            ]
            + [
                nnet.modules.components.DualConvBlock(
                    in_channels=base_channels * 2**i,
                    out_channels=base_channels * 2**i,
                    dropout=dropout,
                )
                for i in range(1, depth)
            ]
        )

        self.downsample = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.dilated_blocks = torch.nn.ModuleList(
            [
                nnet.modules.components.DualConvBlock(
                    in_channels=in_channels,
                    out_channels=base_channels,
                    dilation=2,
                    padding=2,
                    dropout=dropout,
                )
            ]
            + [
                nnet.modules.components.DualConvBlock(
                    in_channels=base_channels * 2 ** (i - 1),
                    out_channels=base_channels * 2**i,
                    dilation=2,
                    padding=2,
                    dropout=dropout,
                )
                for i in range(1, depth - 1)
            ]
        )

        self.se_blocks = torch.nn.ModuleList(
            [
                nnet.modules.components.SEModule(
                    in_channels=base_channels * 2**i,
                    reduction=8,
                )
                for i in range(depth - 1)
            ]
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        enc = self.enc_blocks[0](x)

        features = [enc]

        enc = self.downsample(enc)

        cur_dec = x

        for i, (enc_block, dilated_block, se_block) in enumerate(
            zip(self.enc_blocks[1:], self.dilated_blocks, self.se_blocks)
        ):
            cur_dec = self.downsample(dilated_block(cur_dec))
            se = se_block(cur_dec)
            comb = torch.cat([se, enc], dim=1)
            enc = enc_block(comb)
            features.append(enc)
            enc = self.downsample(enc)

        return features
