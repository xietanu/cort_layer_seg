from dataclasses import dataclass

import numpy as np


@dataclass
class SegmentationOutput:
    labels: np.ndarray
    logits: np.ndarray
    predicted_accuracies: np.ndarray
    depth_maps: np.ndarray | None = None
    input_images: np.ndarray | None = None
    seg_ground_truths: np.ndarray | None = None
    depth_ground_truths: np.ndarray | None = None

    @property
    def num_classes(self) -> int:
        return self.logits.shape[1]

    @property
    def num_samples(self) -> int:
        return self.logits.shape[0]

    @property
    def has_depth_maps(self) -> bool:
        return self.depth_maps is not None

    def __add__(self, other):
        if not isinstance(other, SegmentationOutput):
            raise ValueError(f"Cannot add SegmentationOutput with {type(other)}")

        if (
            self.num_classes != other.num_classes
            or self.has_depth_maps != other.has_depth_maps
        ):
            raise ValueError(
                "Cannot add SegmentationOutputs with different number of classes or depth maps"
            )

        return SegmentationOutput(
            labels=np.concatenate([self.labels, other.labels], axis=0),
            logits=np.concatenate([self.logits, other.logits], axis=0),
            predicted_accuracies=np.concatenate(
                [self.predicted_accuracies, other.predicted_accuracies], axis=0
            ),
            depth_maps=(
                np.concatenate([self.depth_maps, other.depth_maps], axis=0)
                if self.has_depth_maps
                else None
            ),
            input_images=(
                np.concatenate([self.input_images, other.input_images], axis=0)
                if self.input_images is not None
                else None
            ),
            seg_ground_truths=(
                np.concatenate(
                    [self.seg_ground_truths, other.seg_ground_truths], axis=0
                )
                if self.seg_ground_truths is not None
                else None
            ),
            depth_ground_truths=(
                np.concatenate(
                    [self.depth_ground_truths, other.depth_ground_truths], axis=0
                )
                if self.depth_ground_truths is not None
                else None
            ),
        )
