import os
import shutil
import json
import cv2
import numpy as np

import cort.load

PATCH_PATH = "data/all_cort_patches"
BASE_PATH = "nnUNet_raw/Dataset606_CorticalLayers"
IMAGE_FILE = "image.png"
LABEL_FILE = "layermask.png"
TRAIN_FILE = "training.txt"
VAL_FILE = "val.txt"
TEST_FILE = "test.txt"
TRAIN_IMG_FOLDER = "imagesTr"
TRAIN_LABEL_FOLDER = "labelsTr"
TEST_IMG_FOLDER = "imagesTs"
TEST_LABEL_FOLDER = "labelsTs"

DATABASE_CONFIG = {
    "channel_names": {
        "0": "image",
    },
    "labels": {
        "background": 0,
        "I": 1,
        "II": 2,
        "III": 3,
        "IV": 4,
        "V": 5,
        "VI": 6,
        "WM": 7,
    },
    "file_ending": ".png",
}


def main():
    if os.path.exists(BASE_PATH):
        shutil.rmtree(BASE_PATH)

    os.makedirs(BASE_PATH, exist_ok=True)
    os.makedirs(os.path.join(BASE_PATH, TRAIN_IMG_FOLDER), exist_ok=True)
    os.makedirs(os.path.join(BASE_PATH, TRAIN_LABEL_FOLDER), exist_ok=True)
    os.makedirs(os.path.join(BASE_PATH, TEST_IMG_FOLDER), exist_ok=True)
    os.makedirs(os.path.join(BASE_PATH, TEST_LABEL_FOLDER), exist_ok=True)

    all_files = cort.load.find_all_patches(PATCH_PATH)

    test_files = np.random.choice(all_files, len(all_files) // 10, replace=False)
    train_files = [patch for patch in all_files if patch not in test_files]

    print(f"Number of patches: {len(all_files)}")
    print(f"Number of training patches: {len(train_files)}")
    print(f"Number of test patches: {len(test_files)}")

    for i, file in enumerate(train_files):
        file = file.strip()
        image_path = os.path.join(file, IMAGE_FILE)
        label_path = os.path.join(file, LABEL_FILE)
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        img = cv2.resize(img, None, fx=1 / 20, fy=1 / 20, interpolation=cv2.INTER_AREA)
        label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
        label = cv2.resize(
            label, None, fx=1 / 20, fy=1 / 20, interpolation=cv2.INTER_NEAREST
        )
        top_size = label.shape[0] // 2
        label_top = label[:top_size, :]
        label_top[label_top == 7] = 0
        label[:top_size, :] = label_top

        img = img[:, 1:]
        label = label[:, 1:]

        cv2.imwrite(
            os.path.join(BASE_PATH, TRAIN_IMG_FOLDER, f"image_{i:03d}_0000.png"),
            img,
        )
        cv2.imwrite(
            os.path.join(BASE_PATH, TRAIN_LABEL_FOLDER, f"image_{i:03d}.png"),
            label,
        )

    for i, file in enumerate(test_files):
        file = file.strip()
        image_path = os.path.join(file, IMAGE_FILE)
        label_path = os.path.join(file, LABEL_FILE)
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        img = cv2.resize(img, None, fx=1 / 20, fy=1 / 20, interpolation=cv2.INTER_AREA)
        label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
        label = cv2.resize(
            label, None, fx=1 / 20, fy=1 / 20, interpolation=cv2.INTER_NEAREST
        )
        top_size = label.shape[0] // 2
        label_top = label[:top_size, :]
        label_top[label_top == 7] = 0
        label[:top_size, :] = label_top

        img = img[:, 1:]
        label = label[:, 1:]

        cv2.imwrite(
            os.path.join(BASE_PATH, TEST_IMG_FOLDER, f"image_{i:03d}_0000.png"),
            img,
        )
        cv2.imwrite(
            os.path.join(BASE_PATH, TEST_LABEL_FOLDER, f"image_{i:03d}.png"),
            label,
        )

    DATABASE_CONFIG["numTraining"] = len(train_files)

    with open(os.path.join(BASE_PATH, "dataset.json"), "w", encoding="utf-8") as f:
        json.dump(DATABASE_CONFIG, f)

    print("Done!")


if __name__ == "__main__":
    main()
