import numpy as np
import matplotlib.pyplot as plt
import cort.display


def plot_volume(volume: np.ndarray, mask: np.ndarray, title: str = "Volume"):
    fig, axs = plt.subplots(3, 5, figsize=(12, 12))
    for i, j in enumerate(
        range(volume.shape[1] // 10, volume.shape[0], volume.shape[0] // 5)
    ):
        col = cort.display.colour_patch(volume[j, :, :], mask[j, :, :])
        axs[0, i].imshow(col)
        axs[0, i].set_title(f"Layer {j}")

    for i, j in enumerate(
        range(volume.shape[2] // 10, volume.shape[2], volume.shape[2] // 5)
    ):
        col = cort.display.colour_patch(volume[:, :, j].T, mask[:, :, j].T)
        axs[1, i].imshow(col)
        axs[1, i].set_title(f"Layer {j}")

    for i, j in enumerate(
        range(volume.shape[0] // 10, volume.shape[1], volume.shape[1] // 5)
    ):
        col = cort.display.colour_patch(volume[:, j, :], mask[:, j, :])
        axs[2, i].imshow(col)
        axs[2, i].set_title(f"Layer {j}")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
