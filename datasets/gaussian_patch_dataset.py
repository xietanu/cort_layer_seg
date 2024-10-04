import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data
import torchvision

import cort
import datasets.protocols


class GaussianPatchDataset(torch.utils.data.Dataset):
    """A dataset for patches."""

    def __init__(
        self,
        fold: int,
        patches: list[cort.MaskCorticalPatch],
        transform: torchvision.transforms.Compose,
        padded_size: tuple[int, int] | str = "auto",
        provide_position: bool = False,
        condition: datasets.protocols.Condition | None = None,
        img_transform: torchvision.transforms.Compose | None = None,
        roll: float = 0.0,
    ):
        self.fold = fold
        self.patches = patches
        self.transform = transform
        self.img_transform = img_transform
        self.condition = condition
        self.provide_position = provide_position
        self.roll = roll

        if isinstance(padded_size, tuple):
            self.padded_size = padded_size
        elif padded_size == "auto":
            self.padded_size = find_min_padding_size(patches, 16)
        else:
            raise ValueError(f"Invalid padded size argument: {padded_size}")

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int):
        patch = self.patches[idx]

        label = patch.mask
        depth = patch.depth_maps
        prev_mask = patch.mask

        label, depth, prev_mask = [
            center_pad_to_size(arr, self.padded_size)
            for arr in [label, depth, prev_mask]
        ]

        image_label = np.stack((label, prev_mask), axis=-1)
        image_label_depth = np.concatenate((image_label, depth), axis=-1)

        transformed_image_and_label = self.transform(image_label_depth)

        label = transformed_image_and_label[None, 0, :, :]
        prev_mask = transformed_image_and_label[None, 1, :, :]
        depth = transformed_image_and_label[2:, :, :]

        label, depth = [
            center_pad_to_size_tensor(arr, self.padded_size) for arr in [label, depth]
        ]

        if self.roll != 0.0:
            y_roll = int(
                (np.random.randint(0, label.shape[1]) - label.shape[1] // 2) * self.roll
            )
            x_roll = int(
                (np.random.randint(0, label.shape[2]) - label.shape[2] // 2) * self.roll
            )
            label = torch.roll(label, shifts=(y_roll, x_roll), dims=(1, 2))
            depth = torch.roll(depth, shifts=(y_roll, x_roll), dims=(1, 2))
            prev_mask = torch.roll(prev_mask, shifts=(y_roll, x_roll), dims=(1, 2))
            if y_roll > 0:
                label[:, :y_roll, :] = cort.constants.PADDING_MASK_VALUE
                depth[:, :y_roll, :] = cort.constants.PADDING_MASK_VALUE
                prev_mask[:, :y_roll, :] = cort.constants.PADDING_MASK_VALUE
            elif y_roll < 0:
                label[:, y_roll:, :] = cort.constants.PADDING_MASK_VALUE
                depth[:, y_roll:, :] = cort.constants.PADDING_MASK_VALUE
                prev_mask[:, y_roll:, :] = cort.constants.PADDING_MASK_VALUE
            if x_roll > 0:
                label[:, :, :x_roll] = cort.constants.PADDING_MASK_VALUE
                depth[:, :, :x_roll] = cort.constants.PADDING_MASK_VALUE
                prev_mask[:, :, :x_roll] = cort.constants.PADDING_MASK_VALUE
            elif x_roll < 0:
                label[:, :, x_roll:] = cort.constants.PADDING_MASK_VALUE
                depth[:, :, x_roll:] = cort.constants.PADDING_MASK_VALUE
                prev_mask[:, :, x_roll:] = cort.constants.PADDING_MASK_VALUE

        label = label.round().long()

        mask = label == cort.constants.PADDING_MASK_VALUE

        depth[mask.expand_as(depth)] = 0

        image = create_image_from_mask(label)

        inputs = [image]
        if self.condition is not None:
            condition = self.condition(patch)
            if torch.any(torch.isnan(condition)):
                condition = torch.zeros_like(condition)
            inputs.append(condition)
        else:
            inputs.append(torch.tensor(0))
        if self.provide_position:
            inputs.append(torch.tensor([patch.x, patch.y, patch.z]))
        else:
            inputs.append(torch.tensor(0))

        return (
            (
                patch.brain_area,
                patch.section_id,
                patch.patch_id,
                self.fold,
                False,
            ),
            tuple(inputs),
            (label, depth, prev_mask),
        )


def find_min_padding_size(
    patches: list[cort.MaskCorticalPatch], factor: int
) -> tuple[int, int]:
    max_width = np.max([patch.mask.shape[1] for patch in patches])
    max_height = np.max([patch.mask.shape[0] for patch in patches])
    pad_width = int(np.ceil(max_width / factor) * factor)
    pad_height = int(np.ceil(max_height / factor) * factor)
    return pad_height, pad_width


def pad_to_size(arr: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    padded = torch.full(
        (arr.shape[0], *size),
        fill_value=cort.constants.PADDING_MASK_VALUE,
        dtype=torch.float32,
    )
    padded[:, : arr.shape[1], : arr.shape[2]] = arr
    return padded


def center_pad_to_size(arr: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    if arr is None or len(arr.shape) == 0:
        return None
    if arr.ndim == 2:
        arr = arr[:, :, None]
    padded = np.full(
        (*size, arr.shape[2]),
        fill_value=cort.constants.PADDING_MASK_VALUE,
        dtype=arr.dtype,
    )
    h_offset = (size[0] - arr.shape[0]) // 2
    w_offset = (size[1] - arr.shape[1]) // 2
    padded[h_offset : h_offset + arr.shape[0], w_offset : w_offset + arr.shape[1]] = arr
    return padded.squeeze()


def center_pad_to_size_tensor(arr: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    if arr is None or len(arr.shape) == 0:
        return None
    if arr.ndim == 2:
        arr = arr[None, :, :]
    padded = torch.full(
        (arr.shape[0], *size),
        fill_value=cort.constants.PADDING_MASK_VALUE,
        dtype=arr.dtype,
    )
    h_offset = (size[0] - arr.shape[1]) // 2
    w_offset = (size[1] - arr.shape[2]) // 2
    padded[
        :, h_offset : h_offset + arr.shape[1], w_offset : w_offset + arr.shape[2]
    ] = arr
    return padded


def create_image_from_mask(mask: torch.Tensor) -> torch.Tensor:
    """Create an image from a mask."""
    image = torch.normal(mean=0, std=1, size=mask.shape)

    for layer in torch.unique(mask):
        if layer == cort.constants.PADDING_MASK_VALUE:
            continue
        image[mask == layer] += torch.normal(mean=0, std=1, size=(1,)).item()

    image = (image - image.min()) / (image.max() - image.min())
    image[mask == cort.constants.PADDING_MASK_VALUE] = 0

    return image
