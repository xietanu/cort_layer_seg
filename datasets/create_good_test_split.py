import itertools

import numpy as np

import datasets


def create_good_test_split(filepaths: list[str]) -> tuple[list[str], ...]:
    """Split a list of filepaths into different sets"""
    folders = [f.split("\\")[1] for f in filepaths]

    filepaths_by_folder = {}

    for filepath, folder in zip(filepaths, folders):
        if folder not in filepaths_by_folder:
            filepaths_by_folder[folder] = []
        filepaths_by_folder[folder].append(filepath)

    test_samples = []
    val_samples = []

    for folder in filepaths_by_folder:
        filepaths_by_folder[folder] = np.random.permutation(filepaths_by_folder[folder])
        test_samples.append(filepaths_by_folder[folder][:1])
        val_samples.append(filepaths_by_folder[folder][1])
        filepaths_by_folder[folder] = filepaths_by_folder[folder][2:]

    test_samples = list(itertools.chain(*test_samples))

    remaining_samples = list(itertools.chain(*filepaths_by_folder.values()))

    return remaining_samples, val_samples, test_samples
