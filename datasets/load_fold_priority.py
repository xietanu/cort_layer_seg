import json
import torch.utils.data

import cort.load
import cort.manip
import datasets

FOLDER_PATH = "data/preprocessed"
FOLDS_PATH = "data/preprocessed/folds.json"


def load_fold_priority(
    fold: int,
    batch_size: int = 4,
    use_position: bool = False,
    condition: datasets.protocols.Condition | None = None,
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
    train_patches = cort.manip.pad_patches(train_patches, (256, 128))

    print("Loading validation patches...")
    val_patches = cort.load.load_preprocessed_patches(
        FOLDER_PATH, val_filepaths, report_progress=True
    )
    val_patches = cort.manip.pad_patches(val_patches, (256, 128))

    print("Loading test patches...")
    test_patches = cort.load.load_preprocessed_patches(
        FOLDER_PATH, test_filepaths, report_progress=True
    )
    test_patches = cort.manip.pad_patches(test_patches, (256, 128))

    train_dataset = datasets.PatchDataset(
        train_patches,
        datasets.transforms.AUGMENTATIONS,
        condition=condition,
        provide_position=use_position,
    )
    val_dataset = datasets.PatchDataset(
        val_patches,
        datasets.transforms.TO_TENSOR,
        condition=condition,
        provide_position=use_position,
    )
    test_dataset = datasets.PatchDataset(
        test_patches,
        datasets.transforms.TO_TENSOR,
        condition=condition,
        provide_position=use_position,
    )

    train_loader = datasets.PriorityLoader(train_dataset, batch_size=batch_size)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    return datasets.Fold(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        test_dataset=test_dataset,
        test_dataloader=test_loader,
    )
