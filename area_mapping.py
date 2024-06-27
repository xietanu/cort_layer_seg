import json
import os

import numpy as np

import cort
import cort.load


DATA_FOLDER = "data/preprocessed"


def main():
    patch_paths = cort.load.find_all_preprocessed(DATA_FOLDER)
    patches = cort.load.load_preprocessed_patches(
        DATA_FOLDER, patch_paths, report_progress=True
    )

    print(f"Loaded {len(patches)} patches.")

    all_brain_areas = set(patch.brain_area for patch in patches)

    print(f"Found {len(all_brain_areas)} brain areas.")

    one_hot_mapping = {
        brain_area: np.eye(len(all_brain_areas))[i].tolist()
        for i, brain_area in enumerate(all_brain_areas)
    }

    with open(os.path.join(DATA_FOLDER, "area_mapping.json"), "w") as f:
        json.dump(one_hot_mapping, f)

    print("Saved area mapping to area_mapping.json.")


if __name__ == "__main__":
    main()
