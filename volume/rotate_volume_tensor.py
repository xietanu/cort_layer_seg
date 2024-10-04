import numpy as np
import torch


def create_rotation_matrix_tensor(
    x_angle: torch.Tensor, y_angle: torch.Tensor, z_angle: torch.Tensor
) -> torch.Tensor:
    """
    Create a rotation matrix around the x-axis.
    """
    rotation_matrix = (
        create_z_rotation_matrix_tensor(z_angle)
        @ create_y_rotation_matrix_tensor(y_angle)
        @ create_x_rotation_matrix_tensor(x_angle)
    )

    return rotation_matrix


def create_x_rotation_matrix_tensor(angle: torch.Tensor) -> torch.Tensor:
    """
    Create a rotation matrix around the x-axis.
    """
    rotation_matrix = torch.tensor(
        [
            [1, 0, 0, 0],
            [0, torch.cos(angle), -torch.sin(angle), 0],
            [0, torch.sin(angle), torch.cos(angle), 0],
            [0, 0, 0, 1],
        ],
        device="cuda"
    )

    return rotation_matrix


def create_y_rotation_matrix_tensor(angle: torch.Tensor) -> torch.Tensor:
    """
    Create a rotation matrix around the y-axis.
    """
    rotation_matrix = torch.tensor(
        [
            [torch.cos(angle), 0, torch.sin(angle), 0],
            [0, 1, 0, 0],
            [-torch.sin(angle), 0, torch.cos(angle), 0],
            [0, 0, 0, 1],
        ],
        device="cuda"
    )

    return rotation_matrix


def create_z_rotation_matrix_tensor(angle: torch.Tensor) -> torch.Tensor:
    """
    Create a rotation matrix around the z-axis.
    """
    rotation_matrix = torch.tensor(
        [
            [torch.cos(angle), -torch.sin(angle), 0, 0],
            [torch.sin(angle), torch.cos(angle), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        device="cuda"
    )

    return rotation_matrix

def affine(
    voxel_grids: torch.Tensor,
    affine_matrices: torch.Tensor,
    *,
    mode: str = "bilinear",
    padding_mode: str = "border",
):
    """Applies an affine transform to a set of voxels.

    Defaults to returning a torch tensor if the input is a torch tensor,
    and a numpy array if the input is a numpy array. This can be overridden
    by setting `return_torch` to `True` or `False`.
    """
    out_dims = voxel_grids.ndim

    while voxel_grids.ndim < 5:
        voxel_grids = voxel_grids.unsqueeze(0)

    if affine_matrices.shape[-2:] == (4, 4):
        affine_matrices = affine_matrices[..., :3, :4]
    elif affine_matrices.shape[-2:] != (3, 4):
        raise ValueError(
            "Invalid affine matrix shape: "
            + str(affine_matrices.shape)
            + ", expected ([N], 3, 4)"
        )
    if affine_matrices.ndim == 2:
        affine_matrices = affine_matrices.unsqueeze(0)
    elif affine_matrices.ndim != 3:
        raise ValueError(
            "Invalid affine matrix shape: "
            + str(affine_matrices.shape)
            + ", expected ([N], 3, 4)"
        )

    affine_grids = torch.nn.functional.affine_grid(
        affine_matrices,
        list(voxel_grids.shape),
        align_corners=False,
    )

    views = torch.nn.functional.grid_sample(
        voxel_grids,
        affine_grids.double(),
        mode=mode,
        padding_mode=padding_mode,
        align_corners=False,
    )
    views = views.clamp(0, 1)

    while views.ndim > out_dims:
        views = views.squeeze(0)

    return views

def rotated(
    voxel_grid: np.ndarray | torch.Tensor,
    angles: np.ndarray | torch.Tensor,
    padding_mode: str = "border",
) -> np.ndarray | torch.Tensor:
    """Returns a rotated version of the voxel model."""
    # if np.allclose(angles, 0):
    #     return voxel_grid

    if angles.shape[-1] != 3:
        raise ValueError(
            "Invalid angle shape: " + str(angles.shape) + ", expected ([N], 3)"
        )

    rotate_matrices = create_rotation_matrix_tensor(
        angles[0], angles[1], angles[2]
    )

    return affine(
        voxel_grid, rotate_matrices, padding_mode=padding_mode, mode="nearest"
    ), rotate_matrices