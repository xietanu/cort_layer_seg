import os


def find_all_patches(base_path: str) -> list[str]:
    """Find all patches in a base path."""
    patches_to_load = []
    info_file = os.path.join(base_path, "info.txt")
    if os.path.isfile(info_file):
        patches_to_load = [base_path]
    else:
        for folder in os.listdir(base_path):
            folder_path = os.path.join(base_path, folder)
            if os.path.isdir(folder_path):
                patches_to_load.extend(find_all_patches(folder_path))
    return patches_to_load


def find_all_preprocessed(base_path: str) -> list[str]:
    """Find all preprocessed patches in a base path."""
    patches_to_load = [
        file.replace("_data.json", "")
        for file in os.listdir(base_path)
        if file.endswith("_data.json")
    ]

    return patches_to_load
