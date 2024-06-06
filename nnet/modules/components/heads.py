import torch

import nnet.modules.components


class SegmentationHead(torch.nn.Module):
    def __init__(self, in_channels: int, n_classes: int):
        super().__init__()

        self.conv = nnet.modules.components.DualConvBlock(
            in_channels=in_channels,
            out_channels=in_channels,
        )
        self.output = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=n_classes,
            kernel_size=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return self.output(x)

    def conv_params(self):
        return self.conv.parameters()

    def film_params(self):
        return []


class DepthMapHead(torch.nn.Module):
    def __init__(self, in_channels: int, n_borders: int):
        super().__init__()

        self.conv = nnet.modules.components.DualConvBlock(
            in_channels=in_channels,
            out_channels=in_channels,
        )
        self.output = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=n_borders,
            kernel_size=1,
        )
        self.tanh = torch.nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.output(x)
        return self.tanh(x)

    def conv_params(self):
        return self.conv.parameters()

    def film_params(self):
        return []


class AccuracyHead(torch.nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()

        self.conv = torch.nn.Sequential(
            nnet.modules.components.DualConvBlock(in_channels, in_channels),
            torch.nn.AdaptiveMaxPool2d((1, 1)),
        )

        self.flatten = torch.nn.Flatten()
        self.fc = torch.nn.Linear(in_channels, in_channels)
        self.relu = torch.nn.ReLU()
        self.output = torch.nn.Linear(in_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.relu(x)
        return self.output(x)


class BottomAccuracyHead(torch.nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()

        self.conv = torch.nn.Sequential(
            nnet.modules.components.DualConvBlock(in_channels, in_channels),  # 16x8
            torch.nn.AdaptiveAvgPool2d((1, 1)),  # 1x1
        )

        self.flatten = torch.nn.Flatten()
        self.fc = torch.nn.Linear(in_channels, in_channels)
        self.relu = torch.nn.ReLU()
        self.output = torch.nn.Linear(in_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.relu(x)
        return self.output(x)


class DualHead(torch.nn.Module):
    def __init__(self, in_channels: int, n_classes: int):
        super().__init__()

        self.segmentation_head = SegmentationHead(in_channels, n_classes)
        self.depth_map_head = DepthMapHead(in_channels, n_classes - 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.segmentation_head(x), self.depth_map_head(x)

    def conv_params(self):
        return list(self.segmentation_head.conv_params()) + list(
            self.depth_map_head.conv_params()
        )

    def film_params(self):
        return list(self.segmentation_head.film_params()) + list(
            self.depth_map_head.film_params()
        )


class SegAccHead(torch.nn.Module):
    def __init__(self, in_channels: int, n_classes: int):
        super().__init__()

        self.segmentation_head = SegmentationHead(in_channels, n_classes)
        self.accuracy_head = AccuracyHead(in_channels)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.segmentation_head(x), self.accuracy_head(x)


class SegAccDepthHead(torch.nn.Module):
    def __init__(self, in_channels: int, n_classes: int):
        super().__init__()

        self.segmentation_head = SegmentationHead(in_channels, n_classes)
        self.accuracy_head = AccuracyHead(in_channels)
        self.depth_map_head = DepthMapHead(in_channels, n_classes - 1)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.segmentation_head(x), self.accuracy_head(x), self.depth_map_head(x)
