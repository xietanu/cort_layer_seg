from torchvision import transforms

import cort.constants


TO_TENSOR = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

AUGMENTATIONS = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(
        #    10,
        #    fill=cort.constants.PADDING_MASK_VALUE,
        #    interpolation=transforms.InterpolationMode.NEAREST,
        # ),
        transforms.RandomApply(
            [
                transforms.RandomAffine(
                    degrees=10,
                    translate=(0, 0.1),
                    shear=10,
                    fill=cort.constants.PADDING_MASK_VALUE,
                    interpolation=transforms.InterpolationMode.NEAREST,
                ),
            ],
            p=0.85,
        ),
        # transforms.RandomPerspective(
        #    0.1,
        #    fill=cort.constants.PADDING_MASK_VALUE,
        #    interpolation=transforms.InterpolationMode.NEAREST_EXACT,
        # ),
        # transforms.RandomResizedCrop((256, 128), scale=(0.8, 1.0), ratio=(0.75, 1.333)),
    ]
)

AUGMENTATIONS_IMG_ONLY = transforms.Compose(
    [
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
    ]
)
