import itertools

import numpy as np

import datasets


def create_split(
    filepaths: list[str], splits: tuple[float, ...]
) -> tuple[list[str], ...]:
    """Split a list of filepaths into different sets"""
    if not np.isclose(sum(splits), 1):
        raise ValueError("Splits must sum to 1")

    n_files = len(filepaths)

    filepaths = np.random.permutation(filepaths)

    split_points = np.cumsum(
        [0] + list(np.round(np.array(splits) * n_files).astype(int))
    )

    split_sets = [
        filepaths[start:end].tolist() for start, end in itertools.pairwise(split_points)
    ]

    return tuple(split_sets)
