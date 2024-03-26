import cort
import cort.load
import cort.manip
import datasets

SPLIT_DATA_FILEPATHS = {
    datasets.enums.Split.TRAIN: "data/cort_patches/train.txt",
    datasets.enums.Split.VALID: "data/cort_patches/val.txt",
    datasets.enums.Split.TEST: "data/cort_patches/test.txt",
}


def load_split_patches(
    split: datasets.enums.Split, downscale_factor: float, pad: bool = True
) -> list[cort.CorticalPatch]:
    """Load patches for a split."""
    split_data_path = SPLIT_DATA_FILEPATHS[split]

    with open(split_data_path, "r", encoding="utf-8") as f:
        filepaths = f.read().splitlines()

    patches = cort.load.load_patches(
        filepaths, downscale=downscale_factor, report_progress=True
    )

    if pad:
        patches = [cort.manip.pad_patch(patch, (256, 128)) for patch in patches]

    return patches
