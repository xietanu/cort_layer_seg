import json
import torch.utils.data

import cort.load
import cort.manip
import datasets

FOLDER_PATH = "data/preprocessed"
FOLDS_PATH = "data/preprocessed/folds.json"


def load_fold(
    fold: int,
    use_position: bool = False,
    batch_size: int = 4,
    condition: datasets.protocols.Condition | None = None,
    perc_siibra: float = 0.5,
    roll: float = 0.75,
    use_transforms: bool = True,
    use_prev_seg_with_siibra: bool = False,
) -> datasets.Fold:
    """Load training, validation, and test dataloaders for a fold_data."""

    fold_filepaths = json.load(open(FOLDS_PATH))["splits"]
    test_filepaths = fold_filepaths[fold]

    remaining_filepaths = [
        fp for fold in fold_filepaths[:fold] + fold_filepaths[fold + 1 :] for fp in fold
    ]

    n_val = max(10, len(remaining_filepaths) // 10)

    val_filepaths = remaining_filepaths[:n_val]
    train_filepaths = remaining_filepaths[n_val:]

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

    train_dataset = datasets.PatchDataset(
        fold,
        train_patches,
        (
            datasets.transforms.AUGMENTATIONS
            if use_transforms
            else datasets.transforms.TO_TENSOR
        ),
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
        condition=condition,
        provide_position=use_position,
        use_prev_seg_with_siibra=use_prev_seg_with_siibra,
    )
    siibra_test_dataset = datasets.PatchDataset(
        fold,
        test_patches,
        datasets.transforms.TO_TENSOR,
        condition=condition,
        provide_position=use_position,
        percent_siibra=1.0,
        use_prev_seg_with_siibra=use_prev_seg_with_siibra,
    )

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

    return datasets.Fold(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        test_dataset=test_dataset,
        test_dataloader=test_loader,
        siibra_test_dataset=siibra_test_dataset,
        siibra_test_dataloader=siibra_test_loader,
    )
