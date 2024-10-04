import os
import random
import json

import cort.load


def create_alt_folds(base_path: str, n_splits: int = 5) -> None:
    """Create folds for cross-validation."""
    filepaths = cort.load.find_all_preprocessed(base_path)

    grouped_filepaths = {}
    for filepath in filepaths:
        group = filepath.split("-")[0]
        if "corner" in filepath:
            group = "corner"
        if group not in grouped_filepaths:
            grouped_filepaths[group] = []

        grouped_filepaths[group].append(filepath)

    test_filepaths = []

    for group, group_filepaths in grouped_filepaths.items():
        random.shuffle(group_filepaths)
        if group == "corner":
            test_filepaths.extend(group_filepaths[:4])
            grouped_filepaths[group] = group_filepaths[4:]
        else:
            test_filepaths.append(group_filepaths[0])
            grouped_filepaths[group] = group_filepaths[1:]

    train_filepaths = []
    for group_filepaths in grouped_filepaths.values():
        train_filepaths.extend(group_filepaths)

    random.shuffle(train_filepaths)

    split_sizes = [len(train_filepaths) // n_splits for _ in range(n_splits)]

    for i in range(len(train_filepaths) % n_splits):
        split_sizes[i] += 1

    split_dict = {
        "splits": [
            train_filepaths[sum(split_sizes[:i]) : sum(split_sizes[: i + 1])]
            for i in range(n_splits)
        ],
        "test": test_filepaths,
    }

    with open(os.path.join(base_path, "alt_folds.json"), "w") as f:
        json.dump(split_dict, f)
