import os
import random
import json

import cort.load


def create_folds(base_path: str, n_splits: int = 5) -> None:
    """Create folds for cross-validation."""
    filepaths = cort.load.find_all_preprocessed(base_path)

    random.shuffle(filepaths)

    split_sizes = [len(filepaths) // n_splits for _ in range(n_splits)]

    for i in range(len(filepaths) % n_splits):
        split_sizes[i] += 1

    split_dict = {
        "splits": [
            filepaths[sum(split_sizes[:i]) : sum(split_sizes[: i + 1])]
            for i in range(n_splits)
        ]
    }

    with open(os.path.join(base_path, "folds.json"), "w") as f:
        json.dump(split_dict, f)
