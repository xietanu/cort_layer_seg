import numpy as np
import torch

import cort


def unpad_img(
    img: torch.Tensor,
    mask: torch.Tensor,
    padding_value=cort.constants.PADDING_MASK_VALUE,
) -> torch.Tensor:
    """Unpad an image using a mask."""

    non_pad_cols = np.any(mask.numpy() != padding_value, axis=1)[0]
    non_pad_rows = np.any(mask.numpy() != padding_value, axis=2)[0]

    img = img[:, non_pad_rows, :][:, :, non_pad_cols]

    return img


def unpad_mask(
    mask: torch.Tensor,
    padding_value=cort.constants.PADDING_MASK_VALUE,
) -> torch.Tensor:
    """Unpad a mask."""

    return unpad_img(mask, mask, padding_value)
