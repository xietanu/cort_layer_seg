import numpy as np
import torch
from torchvision import transforms
import perlin_noise

import cort.constants
from tqdm import tqdm

TO_TENSOR = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

RANDOM_ROTATION = transforms.RandomApply(
    [
        transforms.RandomRotation(
            45,
            fill=cort.constants.PADDING_MASK_VALUE,
            interpolation=transforms.InterpolationMode.NEAREST,
        )
    ],
    1,
)
RANDOM_PERSPECTIVE = transforms.RandomPerspective(
    0.5,
    p=0.25,
    fill=cort.constants.PADDING_MASK_VALUE,
    interpolation=transforms.InterpolationMode.NEAREST,
)
ELASTIC_TRANSFORM = transforms.RandomApply(
    [
        transforms.ElasticTransform(
            alpha=50.0,
            sigma=5.0,
            fill=cort.constants.PADDING_MASK_VALUE,
            interpolation=transforms.InterpolationMode.NEAREST,
        )
    ],
    0.25,
)
RANDOM_CROP = transforms.RandomApply(
    [
        transforms.RandomCrop(
            (64, 64),
            pad_if_needed=True,
            padding_mode="constant",
            fill=cort.constants.PADDING_MASK_VALUE,
        )
    ],
    1.0,
)

GAUSSIAN_BLUR = transforms.RandomChoice(
    [
        transforms.GaussianBlur(3),
        transforms.GaussianBlur(5),
        transforms.Lambda(lambda x: x),
    ],
    p=[0.125, 0.125, 0.75],
)

ADJUST_GAMMA = transforms.Compose(
    [
        transforms.Lambda(lambda x: x ** (np.random.normal(1, 0.125))),
    ]
)

NOISE = np.load("perl_noise.npy")


def add_perlin_noise(x: torch.Tensor) -> torch.Tensor:
    noise = NOISE[np.random.choice(len(NOISE))]
    amount = np.random.uniform(0.0, 0.5)

    h_offset = np.random.randint(0, noise.shape[0] - x.shape[1])
    w_offset = np.random.randint(0, noise.shape[1] - x.shape[2])
    noise_section = noise[
        h_offset : h_offset + x.shape[1], w_offset : w_offset + x.shape[2]
    ]
    return torch.clamp(x + (amount * noise_section)[None, :, :], 0, 1)


AUGMENTATIONS = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5),
        RANDOM_ROTATION,
        RANDOM_PERSPECTIVE,
        ELASTIC_TRANSFORM,
        # RANDOM_CROP,
    ]
)

DENOISE_AUGMENTATIONS = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(p=0.5),
        RANDOM_ROTATION,
        # RANDOM_PERSPECTIVE,
        # ELASTIC_TRANSFORM,
        # RANDOM_CROP,
    ]
)


AUGMENTATIONS_IMG_ONLY = transforms.Compose(
    [
        GAUSSIAN_BLUR,
        # transforms.ColorJitter(brightness=0.1, contrast=0.1),
        # ADJUST_GAMMA,
        transforms.Lambda(add_perlin_noise),
    ]
)
