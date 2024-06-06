import numpy as np
import torch
import torch.utils.data
import torchvision

import cort
import datasets.protocols


class PatchDataset(torch.utils.data.Dataset):
    """A dataset for patches."""

    def __init__(
        self,
        fold: int,
        patches: list[cort.CorticalPatch],
        transform: torchvision.transforms.Compose,
        padded_size: tuple[int, int] | str = "auto",
        provide_position: bool = False,
        condition: datasets.protocols.Condition | None = None,
        img_transform: torchvision.transforms.Compose | None = None,
        percent_siibra: float = 0.0,
        roll: float = 0.0,
    ):
        self.fold = fold
        self.patches = patches
        self.transform = transform
        self.img_transform = img_transform
        self.condition = condition
        self.provide_position = provide_position
        self.percent_siibra = percent_siibra
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

        if np.random.rand() < self.percent_siibra:
            image = patch.siibra_image
            label = patch.siibra_mask
            depth = patch.siibra_depth_maps
        else:
            image = patch.image
            label = patch.mask
            depth = patch.depth_maps

        image, label, depth = [
            center_pad_to_size(arr, self.padded_size) for arr in [image, label, depth]
        ]

        image_label = np.stack((image, label), axis=-1)

        image_label_depth = np.concatenate((image_label, depth), axis=-1)

        transformed_image_and_label = self.transform(image_label_depth)

        image = transformed_image_and_label[None, 0, :, :]
        label = transformed_image_and_label[None, 1, :, :]
        depth = transformed_image_and_label[2:, :, :]

        image, label, depth = [
            center_pad_to_size_tensor(arr, self.padded_size)
            for arr in [image, label, depth]
        ]

        if self.roll != 0.0:
            y_roll = int(
                (np.random.randint(0, image.shape[1]) - image.shape[1] // 2) * self.roll
            )
            x_roll = int(
                (np.random.randint(0, image.shape[2]) - image.shape[2] // 2) * self.roll
            )
            image = torch.roll(image, shifts=(y_roll, x_roll), dims=(1, 2))
            label = torch.roll(label, shifts=(y_roll, x_roll), dims=(1, 2))
            depth = torch.roll(depth, shifts=(y_roll, x_roll), dims=(1, 2))
            if y_roll > 0:
                image[:, :y_roll, :] = cort.constants.PADDING_MASK_VALUE
                label[:, :y_roll, :] = cort.constants.PADDING_MASK_VALUE
                depth[:, :y_roll, :] = cort.constants.PADDING_MASK_VALUE
            elif y_roll < 0:
                image[:, y_roll:, :] = cort.constants.PADDING_MASK_VALUE
                label[:, y_roll:, :] = cort.constants.PADDING_MASK_VALUE
                depth[:, y_roll:, :] = cort.constants.PADDING_MASK_VALUE
            if x_roll > 0:
                image[:, :, :x_roll] = cort.constants.PADDING_MASK_VALUE
                label[:, :, :x_roll] = cort.constants.PADDING_MASK_VALUE
                depth[:, :, :x_roll] = cort.constants.PADDING_MASK_VALUE
            elif x_roll < 0:
                image[:, :, x_roll:] = cort.constants.PADDING_MASK_VALUE
                label[:, :, x_roll:] = cort.constants.PADDING_MASK_VALUE
                depth[:, :, x_roll:] = cort.constants.PADDING_MASK_VALUE

        label = label.round().long()

        mask = label == cort.constants.PADDING_MASK_VALUE

        depth[mask.expand_as(depth)] = 0
        image[image == cort.constants.PADDING_MASK_VALUE] = 0

        if self.img_transform is not None:
            image = self.img_transform(image)

        inputs = [image]
        if self.condition is not None:
            condition = self.condition(patch)
            inputs.append(condition)
        else:
            inputs.append(torch.tensor(0))
        if self.provide_position:
            inputs.append(torch.tensor([patch.x, patch.y, patch.z]))
        else:
            inputs.append(torch.tensor(0))

        return (
            (patch.brain_area, patch.section_id, patch.patch_id, self.fold),
            tuple(inputs),
            (label, depth),
        )


def find_min_padding_size(
    patches: list[cort.CorticalPatch], factor: int
) -> tuple[int, int]:
    max_width = np.max(
        [max(patch.image.shape[1], patch.siibra_image.shape[1]) for patch in patches]
    )
    max_height = np.max(
        [max(patch.image.shape[0], patch.siibra_image.shape[0]) for patch in patches]
    )
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
