import os

import cort.load
import datasets


def create_tvt_splits(
    base_path: str, splits: tuple[float, float, float] = (0.8, 0.1, 0.1)
) -> None:
    """Create training, validation, and test splits."""
    filepaths = cort.load.find_all_patches(base_path)

    # splits = datasets.create_split(filepaths, splits)
    splits = datasets.create_good_test_split(filepaths)

    with open(os.path.join(base_path, "training.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(splits[0]))

    with open(os.path.join(base_path, "val.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(splits[1]))

    with open(os.path.join(base_path, "test.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(splits[2]))
