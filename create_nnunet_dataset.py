import os
import shutil
import json
import cv2


PATCH_PATH = "data/cort_patches"
BASE_PATH = "nnUNet_raw/Dataset606_CorticalLayers"
IMAGE_FILE = "image.png"
LABEL_FILE = "layermask.png"
TRAIN_FILE = "train.txt"
VAL_FILE = "val.txt"
TEST_FILE = "test.txt"
TRAIN_IMG_FOLDER = "imagesTr"
TRAIN_LABEL_FOLDER = "labelsTr"
TEST_IMG_FOLDER = "imagesTs"

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

    with open(os.path.join(PATCH_PATH, TRAIN_FILE), "r", encoding="utf-8") as f:
        train_files = f.readlines()

    with open(os.path.join(PATCH_PATH, VAL_FILE), "r", encoding="utf-8") as f:
        val_files = f.readlines()

    with open(os.path.join(PATCH_PATH, TEST_FILE), "r", encoding="utf-8") as f:
        test_files = f.readlines()

    train_files = train_files + val_files

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
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        img = cv2.resize(img, None, fx=1 / 20, fy=1 / 20, interpolation=cv2.INTER_AREA)
        cv2.imwrite(
            os.path.join(BASE_PATH, TEST_IMG_FOLDER, f"image_{i:03d}_0000.png"),
            img,
        )

    DATABASE_CONFIG["numTraining"] = len(train_files)

    with open(os.path.join(BASE_PATH, "dataset.json"), "w", encoding="utf-8") as f:
        json.dump(DATABASE_CONFIG, f)

    print("Done!")


if __name__ == "__main__":
    main()
