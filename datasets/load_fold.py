import json
import torch.utils.data

import cort.load
import cort.manip
import datasets

FOLDER_PATH = "data/preprocessed"
RANDOM_FOLDER_PATH = "data/preprocessed_random"
FOLDS_PATH = "data/preprocessed/alt_folds.json"
GUASSIAN_FOLDER_PATH = "data/preprocessed_gaussian"


def load_fold(
    fold: int,
    use_position: bool = False,
    batch_size: int = 4,
    condition: datasets.protocols.Condition | None = None,
    perc_siibra: float = 0.5,
    roll: float = 0.75,
    use_transforms: bool = True,
    use_prev_seg_with_siibra: bool = False,
    padded_size: tuple[int, int] | None = None,
    use_random: bool = False,
    use_gaussian: bool = False,
    use_synth: bool = False,
) -> datasets.Fold:
    """Load training, validation, and test dataloaders for a fold_data."""
    if padded_size is None:
        padded_size = "auto"

    fold_data = json.load(open(FOLDS_PATH))
    fold_filepaths = fold_data["splits"]
    val_filepaths = fold_filepaths[fold]
    test_filepaths = fold_data["test"]

    train_filepaths = [
        fp for fold in fold_filepaths[:fold] + fold_filepaths[fold + 1 :] for fp in fold
    ]

    print("Loading training patches...")
    train_patches = cort.load.load_preprocessed_patches(
        FOLDER_PATH, train_filepaths, report_progress=True
    )
    # train_patches = cort.manip.pad_patches(train_patches, (256, 128))

    print("Loading validation patches...")
    val_patches = cort.load.load_preprocessed_patches(
        FOLDER_PATH, val_filepaths, report_progress=True
    )
    # val_patches = cort.manip.pad_patches(val_patches, (256, 128))

    print("Loading test patches...")
    test_patches = cort.load.load_preprocessed_patches(
        FOLDER_PATH, test_filepaths, report_progress=True
    )
    # test_patches = cort.manip.pad_patches(test_patches, (256, 128))

    if use_random:
        print("Loading random patches...")
        random_patches = cort.load.load_preprocessed_patches(
            RANDOM_FOLDER_PATH, report_progress=True
        )
    else:
        random_patches = None

    if use_gaussian:
        print("Loading gaussian patches...")
        gaussian_patches = cort.load.load_pre_mask_patches(
            GUASSIAN_FOLDER_PATH, report_progress=True
        )
    else:
        gaussian_patches = None

    if use_synth:
        print("Loading synth patches...")
        synth_patches = cort.load.load_preprocessed_patches(
            "data/synth", report_progress=True
        )

    train_dataset = datasets.PatchDataset(
        fold,
        train_patches,
        (
            datasets.transforms.AUGMENTATIONS
            if use_transforms
            else datasets.transforms.TO_TENSOR
        ),
        padded_size=padded_size,
        condition=condition,
        provide_position=use_position,
        img_transform=(
            datasets.transforms.AUGMENTATIONS_IMG_ONLY if use_transforms else None
        ),
        percent_siibra=perc_siibra,
        roll=roll,
        use_prev_seg_with_siibra=use_prev_seg_with_siibra,
    )
    val_dataset = datasets.PatchDataset(
        fold,
        val_patches,
        datasets.transforms.TO_TENSOR,
        padded_size=padded_size,
        condition=condition,
        provide_position=use_position,
        percent_siibra=perc_siibra,
        roll=0.0,
        use_prev_seg_with_siibra=use_prev_seg_with_siibra,
    )
    test_dataset = datasets.PatchDataset(
        fold,
        test_patches,
        datasets.transforms.TO_TENSOR,
        padded_size=padded_size,
        condition=condition,
        provide_position=use_position,
        use_prev_seg_with_siibra=use_prev_seg_with_siibra,
    )
    siibra_test_dataset = datasets.PatchDataset(
        fold,
        test_patches,
        datasets.transforms.TO_TENSOR,
        padded_size=padded_size,
        condition=condition,
        provide_position=use_position,
        percent_siibra=1.0,
        use_prev_seg_with_siibra=use_prev_seg_with_siibra,
    )

    if use_random:
        random_dataset = datasets.PatchDataset(
            fold,
            random_patches,
            datasets.transforms.TO_TENSOR,
            padded_size=padded_size,
            condition=condition,
            provide_position=use_position,
            percent_siibra=0.0,
            roll=0.0,
        )
    else:
        random_dataset = None

    if use_gaussian:
        gaussian_dataset = datasets.GaussianPatchDataset(
            fold,
            gaussian_patches,
            datasets.transforms.TO_TENSOR,
            padded_size=padded_size,
            condition=condition,
            provide_position=use_position,
            roll=0.0,
        )
    else:
        gaussian_dataset = None

    if use_synth:
        synth_dataset = datasets.PatchDataset(
            fold,
            synth_patches,
            datasets.transforms.TO_TENSOR,
            padded_size=padded_size,
            condition=condition,
            provide_position=use_position,
            percent_siibra=0.0,
            roll=0.0,
            use_prev_seg_with_siibra=use_prev_seg_with_siibra,
        )
    else:
        synth_dataset = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )
    siibra_test_loader = torch.utils.data.DataLoader(
        siibra_test_dataset, batch_size=batch_size, shuffle=False
    )
    if use_random:
        random_loader = torch.utils.data.DataLoader(
            random_dataset, batch_size=1, shuffle=True
        )
    else:
        random_loader = None

    if use_gaussian:
        gaussian_loader = torch.utils.data.DataLoader(
            gaussian_dataset, batch_size=batch_size, shuffle=True
        )
    else:
        gaussian_loader = None

    if use_synth:
        synth_loader = torch.utils.data.DataLoader(
            synth_dataset, batch_size=batch_size, shuffle=True
        )
    else:
        synth_loader = None

    return datasets.Fold(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        test_dataset=test_dataset,
        test_dataloader=test_loader,
        siibra_test_dataset=siibra_test_dataset,
        siibra_test_dataloader=siibra_test_loader,
        random_dataloader=random_loader,
        gaussian_dataloader=gaussian_loader,
        synth_dataloader=synth_loader,
    )
