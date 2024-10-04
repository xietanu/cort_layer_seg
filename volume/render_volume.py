import timeit

import cv2
import numpy as np


def render_volume(
    coloured_volume: np.ndarray,
    cutout: tuple[float, float, float] = (0.5, 0.5, 0.5),
    include_bg: bool = True,
) -> np.ndarray:
    start_time = timeit.default_timer()

    min_width = min(coloured_volume.shape[0], coloured_volume.shape[2])
    col_vol = coloured_volume[:min_width, :, :min_width].copy()
    col_vol = col_vol[2:-2, 2:-2, 2:-2]

    vol_width = col_vol.shape[0]
    vol_height = col_vol.shape[1]
    vol_depth = col_vol.shape[2]

    cutout = (np.array(cutout) * col_vol.shape[:3]).astype(int)

    col_vol[-cutout[0] :, : cutout[1], : cutout[2], -1] = 0

    alpha_mask = col_vol[:, :, :, 3] > 0.1

    light_mask = np.logical_xor(alpha_mask, np.roll(alpha_mask, -1, axis=0))
    light_mask[-1, :, :] = True
    light_mask[0, :, :] = False
    col_vol[light_mask, :3] *= 0.85

    dark_mask_1 = np.logical_xor(alpha_mask, np.roll(alpha_mask, 1, axis=2))
    dark_mask_2 = np.logical_xor(alpha_mask, np.roll(alpha_mask, 1, axis=0))
    dark_mask = np.logical_or(dark_mask_1, dark_mask_2)
    dark_mask[:, :, -1] = False
    dark_mask[:, :, 0] = True
    col_vol[dark_mask, :3] *= 1.2

    new_shape = int(np.ceil(col_vol.shape[2] * np.sqrt(2))), vol_height + int(
        np.ceil(col_vol.shape[0] * np.sqrt(2))
    )

    rot_45 = cv2.getRotationMatrix2D(
        (col_vol.shape[2] // 2, col_vol.shape[0] // 2), 45, 1
    )
    rot_45[:, 2] += col_vol.shape[0] * (np.sqrt(2) - 1) / 2
    # rot_45[1, 2] += vol_height

    stack = col_vol.transpose(0, 2, 1, 3).reshape(vol_width, vol_depth, -1)

    rot_stack = cv2.warpAffine(
        stack,
        rot_45,
        new_shape,
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).reshape(new_shape[1], new_shape[0], vol_depth, 4)

    print(rot_stack.shape)

    img = np.zeros((new_shape[1], new_shape[0], 4))

    for i in reversed(range(0, vol_height)):
        rotated = rot_stack[:, :, i, :].roll(i, axis=1)
        # cur_slice = np.clip(col_vol[:, i, :], 0, 1)
        # if np.all(cur_slice[:, :, -1] < 0.1):
        #    break

        # rotated = cv2.warpAffine(
        #    cur_slice,
        #    rot_45,
        #    new_shape,
        #    flags=cv2.INTER_NEAREST,
        #    borderMode=cv2.BORDER_CONSTANT,
        #    borderValue=(0, 0, 0, 0),
        # )
        img[rotated[:, :, -1] > 0, :] = rotated[rotated[:, :, -1] > 0]

        # rot_45[1, 2] -= 1

    if include_bg:
        bg = np.ones_like(img)
        bg[img[:, :, -1] > 0, :] = img[img[:, :, -1] > 0]
        img = bg

    img = cv2.resize(
        img, (img.shape[1], int(img.shape[0] * 0.7)), interpolation=cv2.INTER_AREA
    )
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)

    print(f"Rendered volume in {timeit.default_timer() - start_time:.2f}s")

    return img


def precompute_render(coloured_volume: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    min_width = min(coloured_volume.shape[0], coloured_volume.shape[2])
    col_vol = coloured_volume[:min_width, :, :min_width].copy()
    col_vol = col_vol[2:-2, 2:-2, 2:-2]

    alpha_mask = col_vol[:, :, :, 3] > 0.1

    light_mask = np.logical_xor(alpha_mask, np.roll(alpha_mask, -1, axis=0))
    light_mask[-1, :, :] = True
    light_mask[0, :, :] = False
    col_vol[light_mask, :3] *= 0.7

    dark_mask_1 = np.logical_xor(alpha_mask, np.roll(alpha_mask, 1, axis=2))
    dark_mask_2 = np.logical_xor(alpha_mask, np.roll(alpha_mask, 1, axis=0))
    dark_mask = np.logical_or(dark_mask_1, dark_mask_2)
    dark_mask[:, :, -1] = False
    dark_mask[:, :, 0] = True
    col_vol[dark_mask, :3] *= 1.2

    vol_width = col_vol.shape[0]
    vol_height = col_vol.shape[1]
    vol_depth = col_vol.shape[2]

    cutout = np.ones((vol_width, vol_height, vol_depth)).astype(np.uint8)
    cutout[0, :, :] = 2
    cutout[:, :, -1] = 3

    rot_stack = []
    cutout_stack = []

    new_shape = int(np.ceil(col_vol.shape[2] * np.sqrt(2))), vol_height + int(
        np.ceil(col_vol.shape[0] * np.sqrt(2))
    )

    rot_45 = cv2.getRotationMatrix2D(
        (col_vol.shape[2] // 2, col_vol.shape[0] // 2), 45, 1
    )
    rot_45[:, 2] += col_vol.shape[0] * (np.sqrt(2) - 1) / 2
    rot_45[1, 2] += vol_height

    for i in reversed(range(0, vol_height)):
        slice = col_vol[:, i, :]
        cutout_slice = cutout[:, i, :]

        slice = cv2.warpAffine(slice, rot_45, new_shape, flags=cv2.INTER_LINEAR)
        cutout_slice = cv2.warpAffine(
            cutout_slice, rot_45, new_shape, flags=cv2.INTER_NEAREST, borderValue=0
        )

        rot_stack.append(slice)
        cutout_stack.append(cutout_slice)
        rot_45[1, 2] -= 1

    rot_stack = np.stack(rot_stack, axis=1)
    cutout_stack = np.stack(cutout_stack, axis=1)
    cutout_stack = np.pad(
        cutout_stack,
        (
            (cutout_stack.shape[0] // 2, cutout_stack.shape[0] // 2),
            (0, cutout_stack.shape[1]),
            (cutout_stack.shape[2] // 2, cutout_stack.shape[2] // 2),
        ),
    )

    return rot_stack, cutout_stack


def render_precomputed(
    rot_stack,
    cutout_stack,
    col_vol_shape,
    cutout: tuple[float, float, float] = (0.5, 0.5, 0.5),
):
    cutout_vals = (np.array(cutout) * col_vol_shape[:3]).astype(int)
    cutout_vals[0] *= 0.7071
    cutout_vals[2] *= 0.7071

    cutout_stack = np.roll(cutout_stack, cutout_vals[1], axis=1)
    cutout_stack = np.roll(
        cutout_stack, -cutout_vals[1] + cutout_vals[0] + cutout_vals[2], axis=0
    )
    cutout_stack = np.roll(cutout_stack, cutout_vals[0] - cutout_vals[2], axis=2)
    cutout_width = rot_stack.shape[0]
    cutout_height = rot_stack.shape[1]
    cutout_depth = rot_stack.shape[2]
    cutout_stack = cutout_stack[
        cutout_width // 2 : cutout_width // 2 + cutout_width,
        :cutout_height,
        cutout_depth // 2 : cutout_depth // 2 + cutout_depth,
    ]

    rot_stack[cutout_stack == 1, 3] = 0
    rot_stack[cutout_stack == 2, :3] *= 0.85
    rot_stack[cutout_stack == 3, :3] *= 1.2

    front = rot_stack.shape[1] - np.argmax(rot_stack[:, ::-1, :, 3], axis=1) - 1

    x, z = np.indices(front.shape)
    img = rot_stack[x, front, z]
    img = cv2.resize(
        img, (img.shape[1], int(img.shape[0] * 0.7)), interpolation=cv2.INTER_AREA
    )
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)

    return img
