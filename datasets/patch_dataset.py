import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy
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
        use_prev_seg_with_siibra=False,
    ):
        self.fold = fold
        self.patches = patches
        self.transform = transform
        self.img_transform = img_transform
        self.condition = condition
        self.provide_position = provide_position
        self.percent_siibra = percent_siibra
        self.roll = roll
        self.use_prev_seg_with_siibra = use_prev_seg_with_siibra

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

        has_gt = patch.mask is not None

        if (
            has_gt
            and np.random.rand() < self.percent_siibra
            and patch.siibra_imgs is not None
        ):
            image = patch.siibra_imgs.image
            label = (
                patch.siibra_imgs.mask
                if not self.use_prev_seg_with_siibra
                else patch.siibra_imgs.existing_cort_layers
            )
            depth = patch.siibra_imgs.depth_maps
            prev_mask = patch.siibra_imgs.existing_cort_layers
        else:
            image = patch.image
            label = patch.mask if has_gt else np.zeros_like(image)
            depth = (
                patch.depth_maps
                if has_gt
                else np.zeros((image.shape[0], image.shape[1], 2))
            )
            prev_mask = np.zeros_like(image)

        image = image.astype(np.float32)
        image = (image - image.min()) / (image.max() - image.min())

        if self.roll != 0.0:
            for i in range(np.random.randint(0, 6)):
                blood_vessel = gen_blood_vessel()

                borders = patch.borders[:, :, 1:].sum(axis=2)

                probs = borders.flatten() / borders.sum()
                pos = np.random.choice(len(probs), p=probs)
                x_pos = pos % borders.shape[1]
                y_pos = pos // borders.shape[1]

                x_pos += np.random.randint(-5, 5) - blood_vessel.shape[1] // 2
                y_pos += np.random.randint(-5, 5) - blood_vessel.shape[0] // 2

                x_pos = np.clip(x_pos, 0, image.shape[1] - blood_vessel.shape[1])
                y_pos = np.clip(y_pos, 0, image.shape[0] - blood_vessel.shape[0])

                image[
                    y_pos : y_pos + blood_vessel.shape[0],
                    x_pos : x_pos + blood_vessel.shape[1],
                ] += blood_vessel
                image = np.clip(image, 0, 1)

        if self.img_transform is not None:
            image = self.img_transform(image)
            image = image.numpy().transpose(1, 2, 0)

        image, label, depth, prev_mask = [
            center_pad_to_size(arr, self.padded_size)
            for arr in [image, label, depth, prev_mask]
        ]

        image_label = np.stack((image, label, prev_mask), axis=-1)
        image_label_depth = np.concatenate((image_label, depth), axis=-1)

        transformed_image_and_label = self.transform(image_label_depth)

        image = transformed_image_and_label[None, 0, :, :]
        label = transformed_image_and_label[None, 1, :, :]
        prev_mask = transformed_image_and_label[None, 2, :, :]
        depth = transformed_image_and_label[3:, :, :]

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
            prev_mask = torch.roll(prev_mask, shifts=(y_roll, x_roll), dims=(1, 2))
            if y_roll > 0:
                image[:, :y_roll, :] = cort.constants.PADDING_MASK_VALUE
                label[:, :y_roll, :] = cort.constants.PADDING_MASK_VALUE
                depth[:, :y_roll, :] = cort.constants.PADDING_MASK_VALUE
                prev_mask[:, :y_roll, :] = cort.constants.PADDING_MASK_VALUE
            elif y_roll < 0:
                image[:, y_roll:, :] = cort.constants.PADDING_MASK_VALUE
                label[:, y_roll:, :] = cort.constants.PADDING_MASK_VALUE
                depth[:, y_roll:, :] = cort.constants.PADDING_MASK_VALUE
                prev_mask[:, y_roll:, :] = cort.constants.PADDING_MASK_VALUE
            if x_roll > 0:
                image[:, :, :x_roll] = cort.constants.PADDING_MASK_VALUE
                label[:, :, :x_roll] = cort.constants.PADDING_MASK_VALUE
                depth[:, :, :x_roll] = cort.constants.PADDING_MASK_VALUE
                prev_mask[:, :, :x_roll] = cort.constants.PADDING_MASK_VALUE
            elif x_roll < 0:
                image[:, :, x_roll:] = cort.constants.PADDING_MASK_VALUE
                label[:, :, x_roll:] = cort.constants.PADDING_MASK_VALUE
                depth[:, :, x_roll:] = cort.constants.PADDING_MASK_VALUE
                prev_mask[:, :, x_roll:] = cort.constants.PADDING_MASK_VALUE

        label = label.round().long()

        if has_gt:
            mask = label == cort.constants.PADDING_MASK_VALUE
        else:
            mask = image == cort.constants.PADDING_MASK_VALUE

        depth[mask.expand_as(depth)] = 0
        image[image == cort.constants.PADDING_MASK_VALUE] = 0

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
                patch.is_corner_patch,
            ),
            tuple(inputs),
            (label, depth, prev_mask),
        )


def find_min_padding_size(
    patches: list[cort.CorticalPatch], factor: int
) -> tuple[int, int]:
    max_width = np.max(
        [
            (
                max(patch.image.shape[1], patch.siibra_imgs.image.shape[1])
                if patch.siibra_imgs is not None
                else patch.image.shape[1]
            )
            for patch in patches
        ]
    )
    max_height = np.max(
        [
            (
                max(patch.image.shape[0], patch.siibra_imgs.image.shape[0])
                if patch.siibra_imgs is not None
                else patch.image.shape[0]
            )
            for patch in patches
        ]
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


def gen_blood_vessel(max_size=12):
    line = scipy.stats.norm.pdf(np.linspace(-2, 2, max_size), 0, 1)
    line = line / line.max()
    blood_vessel = line[:, np.newaxis] * line[np.newaxis, :]

    if np.random.rand() < 0.5:
        inner = scipy.stats.norm.pdf(np.linspace(-3, 3, max_size), 0, 1)
        inner = 1.25 * inner / inner.max()
        blood_vessel -= inner[:, np.newaxis] * inner[np.newaxis, :]
        blood_vessel *= 2.5

    if np.random.rand() < 0.5:
        blood_vessel *= -1

    blood_vessel *= np.random.uniform(0.3, 0.6)

    x_scale = np.random.uniform(0.25, 1)
    y_scale = np.random.uniform(0.25, 1)

    resized = cv2.resize(
        blood_vessel, (0, 0), fx=x_scale, fy=y_scale, interpolation=cv2.INTER_LINEAR
    )

    rot_angle = np.random.uniform(0, 2 * np.pi)

    rotated = cv2.warpAffine(
        resized,
        cv2.getRotationMatrix2D(
            (resized.shape[0] // 2, resized.shape[1] // 2), rot_angle, 1
        ),
        (max_size, max_size),
        borderValue=0,
    )
    return rotated
