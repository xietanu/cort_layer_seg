import itertools

import torch

import nnet.modules.components


class ImgRegressor(torch.nn.Module):
    def __init__(
        self,
        input_channels: int,
        encoder_map: list[int],
        dropout: float = 0.0,
        uses_condition: bool = False,
        uses_position: bool = False,
        embed_dim: int = -1,
        hidden_embed_dim: int = -1,
    ):
        super().__init__()

        self.uses_condition = uses_condition
        self.uses_position = uses_position

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
            in_channels=input_channels, out_channels=encoder_map[0]
        )

        self.encoder_layers = torch.nn.Sequential(
            *[
                torch.nn.Sequential(
                    nnet.modules.components.ResNetBlock(
                        in_channels=in_channels, out_channels=in_channels
                    ),
                    nnet.modules.components.ResNetBlock(
                        in_channels=in_channels, out_channels=out_channels
                    ),
                    torch.nn.MaxPool2d(kernel_size=2),
                )
                for in_channels, out_channels in itertools.pairwise(encoder_map)
            ]
        )

        end_channels = encoder_map[-1]

        self.adaptive_pool = torch.nn.AdaptiveAvgPool2d((1, 1))

        flat_dim = end_channels + max([hidden_embed_dim, 0])

        self.flatten = torch.nn.Flatten()

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(flat_dim, flat_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(flat_dim, 1),
            torch.nn.Sigmoid(),
        )

    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor | None = None,
        position: torch.Tensor | None = None,
    ) -> torch.Tensor:
        cond = None
        if self.uses_condition:
            condition = self.cond_embed(condition)
            cond = condition
        if self.uses_position:
            position = self.pos_embed(position)
            cond = position
        if self.uses_condition and self.uses_position:
            cond = condition + position
        x = self.input_conv(x)
        x = self.encoder_layers(x)
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        if cond is not None:
            x = torch.cat([x, cond], dim=1)
        x = self.fc(x)
        return x
