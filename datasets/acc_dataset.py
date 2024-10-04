import numpy as np
import torch
import cv2

import cort.constants
import datasets
import datasets.datatypes


class AccuracyDataset:
    def __init__(
        self,
        folder: str,
        noise_level: float = 0.0,
        use_conditions: bool = False,
        use_aug=True,
    ):
        self.noise_level = noise_level
        self.filtered_results = datasets.datatypes.PatchDataItems.load(folder)
        self.use_conditions = use_conditions
        self.use_aug = use_aug

    def __len__(self):
        return len(self.filtered_results)

    def __getitem__(
        self, index: int
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
    ):
        result = self.filtered_results[index]
        logits = result.prediction.logits.clone()
        probs = result.data_input.area_probability
        location = result.data_input.position
        mask = result.ground_truth.segmentation

        if self.noise_level > 0:
            noise = np.random.normal(
                0,
                1,
                size=(logits.shape[1], logits.shape[2], logits.shape[0]),
            )
            for i in range(3):
                noise = cv2.blur(noise, (9, 9))
            noise = (
                noise
                * logits.std().item()
                / noise.std()
                * self.noise_level
                * np.random.uniform(0.0, 1.0)
            )
            noise -= noise.mean()
            noise = noise.transpose(2, 0, 1)
            logits += torch.from_numpy(noise).float()

            if self.use_aug:
                logits_and_mask = torch.cat([logits, mask], dim=0)
                logits_and_mask = datasets.transforms.DENOISE_AUGMENTATIONS(
                    logits_and_mask
                )
                logits = logits_and_mask[:-1]
                logits[logits == cort.constants.PADDING_MASK_VALUE] = 0
                mask = logits_and_mask[-1]
                mask = mask.round().long()

        if self.use_conditions:
            return (logits, probs, location), mask
        return logits, mask
