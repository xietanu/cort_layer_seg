import os

import cort.load

import cv2

BASE_DIR = "pytorch-CycleGAN-and-pix2pix/datasets/cell_seg_paired"
CELL = "B"
SEG = "A"
TRAIN = "train"
VAL = "val"
TEST = "test"

PATCHES_DIR = "data/preprocessed"


def create_cycle_data():
    """Create a cycle dataset."""
    os.makedirs(BASE_DIR, exist_ok=True)
    for half in [CELL, SEG]:
        for split in [TRAIN, VAL, TEST]:
            os.makedirs(os.path.join(BASE_DIR, half, split), exist_ok=True)

    patches = cort.load.load_preprocessed_patches(PATCHES_DIR, report_progress=True)

    print(f"{len(patches)} patches loaded.")

    n_train_patches = int(len(patches) * 0.8)
    n_val_patches = int(len(patches) * 0.1)
    n_test_patches = len(patches) - n_train_patches - n_val_patches

    train_patches = patches[:n_train_patches]
    val_patches = patches[n_train_patches : n_train_patches + n_val_patches]
    test_patches = patches[n_train_patches + n_val_patches :]

    print(f"Train: {len(train_patches)} patches")
    print(f"Val: {len(val_patches)} patches")
    print(f"Test: {len(test_patches)} patches")

    for split_name, split in [
        (TRAIN, train_patches),
        (VAL, val_patches),
        (TEST, test_patches),
    ]:
        for i, patch in enumerate(split):
            cell_img = patch.image
            cell_img = (cell_img * 255).astype("uint8")
            seg_img = patch.mask
            cell_path = os.path.join(BASE_DIR, CELL, split_name, f"{i}.png")
            seg_path = os.path.join(BASE_DIR, SEG, split_name, f"{i}.png")
            cv2.imwrite(cell_path, cell_img)
            cv2.imwrite(seg_path, seg_img)


if __name__ == "__main__":
    create_cycle_data()
