import numpy as np
import cort
import cort.manip


def max_pad_patches(patches: list[cort.CorticalPatch], round_to_pow2:bool = False) -> list[cort.CorticalPatch]:
    """Pad all patches to the same size."""
    max_height = max(patch.shape[0] for patch in patches)
    max_width = max(patch.shape[1] for patch in patches)
    
    if round_to_pow2:
        max_height = 2**int(np.ceil(np.log2(max_height)))
        max_width = 2**int(np.ceil(np.log2(max_width)))

    return [
        cort.manip.pad_patch(patch, (max_height, max_width)) for patch in patches
    ]
