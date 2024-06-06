from tqdm import tqdm

import cort.load


def main():
    patch_paths = cort.load.find_all_patches("data/all_cort_patches")

    patches = cort.load.load_patches(patch_paths, 20, report_progress=True)

    print(f"Loaded {len(patches)} patches.")

    for patch in tqdm(patches, desc="Saving patches"):
        patch.save("data/preprocessed")

    print("Saved all patches. Done!")


if __name__ == "__main__":
    main()
