import os

import cort.load

import cv2

BASE_DIR = "pytorch-CycleGAN-and-pix2pix/datasets/cell_seg"
TRAIN_CELL = "trainB"
TRAIN_SEG = "trainA"
TEST_CELL = "testB"
TEST_SEG = "testA"

CELL_DIR = "data/random_imgs"
SEG_DIR = "data/random_masks"


def create_cycle_data():
    """Create a cycle dataset."""
    os.makedirs(BASE_DIR, exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, TRAIN_CELL), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, TRAIN_SEG), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, TEST_CELL), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, TEST_SEG), exist_ok=True)

    cell_paths = cort.load.find_all_corner_patches(CELL_DIR)
    seg_paths = cort.load.find_all_corner_patches(SEG_DIR)

    print(f"{len(cell_paths)} cell images found.")
    print(f"{len(seg_paths)} seg images found.")

    cell_imgs = [
        cv2.imread(os.path.join(cell_path, "img.png")) for cell_path in cell_paths
    ]

    seg_imgs = [
        cv2.imread(os.path.join(seg_path, "mask.png")) for seg_path in seg_paths
    ]

    n_train_cell_imgs = int(len(cell_imgs) * 0.5)
    n_train_seg_imgs = int(len(seg_imgs) * 0.5)

    print(f"Train cell images: {n_train_cell_imgs}")
    print(f"Train seg images: {n_train_seg_imgs}")

    train_cell_imgs = cell_imgs[:n_train_cell_imgs]
    train_seg_imgs = seg_imgs[:n_train_seg_imgs]

    test_cell_imgs = cell_imgs[n_train_cell_imgs:]
    test_seg_imgs = seg_imgs[n_train_seg_imgs:]

    save_imgs(train_cell_imgs, os.path.join(BASE_DIR, TRAIN_CELL))
    save_imgs(train_seg_imgs, os.path.join(BASE_DIR, TRAIN_SEG))
    save_imgs(test_cell_imgs, os.path.join(BASE_DIR, TEST_CELL))
    save_imgs(test_seg_imgs, os.path.join(BASE_DIR, TEST_SEG))


def save_imgs(imgs, save_dir):
    """Save images to a directory."""
    for i, img in enumerate(imgs):
        cv2.imwrite(os.path.join(save_dir, f"{i}.png"), img)


if __name__ == "__main__":
    create_cycle_data()
