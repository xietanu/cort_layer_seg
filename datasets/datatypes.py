import os
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.utils.data
import json

import cort.manip


@dataclass
class PatchInfos:
    brain_area: list[str]
    section_id: list[int]
    patch_id: list[int]
    fold: list[int]
    is_corner_patch: list[bool]

    def __add__(self, other):
        if not isinstance(other, PatchInfos):
            raise ValueError("other must be PatchInfos")

        return PatchInfos(
            brain_area=self.brain_area + other.brain_area,
            section_id=self.section_id + other.section_id,
            patch_id=self.patch_id + other.patch_id,
            fold=self.fold + other.fold,
            is_corner_patch=self.is_corner_patch + other.is_corner_patch,
        )

    def save(self, folder: str):
        with open(f"{folder}/patch_infos.json", "w") as f:
            json.dump(
                {
                    "brain_area": self.brain_area,
                    "section_id": self.section_id,
                    "patch_id": self.patch_id,
                    "fold": self.fold,
                    "is_corner_patch": self.is_corner_patch,
                },
                f,
            )

    @classmethod
    def load(cls, folder: str):
        with open(f"{folder}/patch_infos.json", "r") as f:
            data = json.load(f)
        return PatchInfos(
            brain_area=data["brain_area"],
            section_id=data["section_id"],
            patch_id=data["patch_id"],
            fold=data["fold"],
            is_corner_patch=data["is_corner_patch"],
        )


@dataclass
class SegInputs:
    input_images: torch.Tensor
    area_probabilities: torch.Tensor | None = None
    position: torch.Tensor | None = None

    def __post_init__(self):
        if (
            self.area_probabilities is not None
            and self.area_probabilities.shape[0] != self.input_images.shape[0]
        ):
            raise ValueError(
                "area_probabilities must have same number of samples as input_images"
            )
        if (
            self.position is not None
            and self.position.shape[0] != self.input_images.shape[0]
        ):
            raise ValueError(
                "position must have same number of samples as input_images"
            )

    def __len__(self):
        return self.input_images.shape[0]

    def to_device(self, device: torch.device):
        self.input_images = self.input_images.to(device)
        if self.area_probabilities is not None:
            self.area_probabilities = self.area_probabilities.to(device)
        if self.position is not None:
            self.position = self.position.to(device)

    def cpu(self):
        self.to_device(torch.device("cpu"))
        return self

    def detach(self):
        self.input_images = self.input_images.detach()
        if self.area_probabilities is not None:
            self.area_probabilities = self.area_probabilities.detach()
        if self.position is not None:
            self.position = self.position.detach()
        return self

    def __add__(self, other):
        if not isinstance(other, SegInputs):
            raise ValueError("other must be SegInputs")

        return SegInputs(
            input_images=torch.cat([self.input_images, other.input_images], dim=0),
            area_probabilities=(
                torch.cat([self.area_probabilities, other.area_probabilities], dim=0)
                if self.area_probabilities is not None
                else None
            ),
            position=(
                torch.cat([self.position, other.position], dim=0)
                if self.position is not None
                else None
            ),
        )

    def save(self, folder: str):
        self.to_device(torch.device("cpu"))
        torch.save(self.input_images, f"{folder}/input_images.pt")
        if self.area_probabilities is not None:
            torch.save(self.area_probabilities, f"{folder}/area_probabilities.pt")
        if self.position is not None:
            torch.save(self.position, f"{folder}/position.pt")

    @classmethod
    def load(cls, folder: str):
        input_images = torch.load(f"{folder}/input_images.pt")
        area_probabilities = (
            torch.load(f"{folder}/area_probabilities.pt")
            if "area_probabilities.pt" in os.listdir(folder)
            else None
        )
        position = (
            torch.load(f"{folder}/position.pt")
            if "position.pt" in os.listdir(folder)
            else None
        )
        return SegInputs(
            input_images=input_images,
            area_probabilities=area_probabilities,
            position=position,
        )


@dataclass
class SegGroundTruths:
    segmentation: torch.Tensor
    depth_maps: torch.Tensor | None = None
    prev_segmentation: torch.Tensor | None = None

    def __post_init__(self):
        if (
            self.depth_maps is not None
            and self.depth_maps.shape[0] != self.segmentation.shape[0]
        ):
            raise ValueError(
                "depth_maps must have same number of samples as segmentation"
            )
        if self.prev_segmentation is not None:
            if self.prev_segmentation.shape[0] != self.segmentation.shape[0]:
                raise ValueError(
                    "prev_segmentation must have same number of samples as segmentation"
                )

    def __len__(self):
        return self.segmentation.shape[0]

    def to_device(self, device: torch.device):
        self.segmentation = self.segmentation.to(device)
        if self.depth_maps is not None:
            self.depth_maps = self.depth_maps.to(device)
        if self.prev_segmentation is not None:
            self.prev_segmentation = self.prev_segmentation.to(device)

    def cpu(self):
        self.to_device(torch.device("cpu"))
        return self

    def detach(self):
        self.segmentation = self.segmentation.detach()
        if self.depth_maps is not None:
            self.depth_maps = self.depth_maps.detach()
        if self.prev_segmentation is not None:
            self.prev_segmentation = self.prev_segmentation.detach()
        return self

    def __add__(self, other):
        if not isinstance(other, SegGroundTruths):
            raise ValueError("other must be SegGroundTruths")

        return SegGroundTruths(
            segmentation=torch.cat([self.segmentation, other.segmentation], dim=0),
            depth_maps=(
                torch.cat([self.depth_maps, other.depth_maps], dim=0)
                if self.depth_maps is not None
                else None
            ),
            prev_segmentation=(
                torch.cat([self.prev_segmentation, other.prev_segmentation], dim=0)
                if self.prev_segmentation is not None
                else None
            ),
        )

    def save(self, folder: str):
        self.to_device(torch.device("cpu"))
        torch.save(self.segmentation, f"{folder}/gt_segmentation.pt")
        if self.depth_maps is not None:
            torch.save(self.depth_maps, f"{folder}/gt_depth_maps.pt")
        if self.prev_segmentation is not None:
            torch.save(self.prev_segmentation, f"{folder}/gt_prev_segmentation.pt")

    @classmethod
    def load(cls, folder: str):
        segmentation = torch.load(f"{folder}/gt_segmentation.pt")
        depth_maps = (
            torch.load(f"{folder}/gt_depth_maps.pt")
            if "gt_depth_maps.pt" in os.listdir(folder)
            else None
        )
        prev_segmentation = (
            torch.load(f"{folder}/gt_prev_segmentation.pt")
            if "gt_prev_segmentation.pt" in os.listdir(folder)
            else None
        )
        return SegGroundTruths(
            segmentation=segmentation,
            depth_maps=depth_maps,
            prev_segmentation=prev_segmentation,
        )


@dataclass
class Predictions:
    segmentation: torch.Tensor
    logits: torch.Tensor
    accuracy: torch.Tensor
    depth_maps: torch.Tensor | None = None
    denoised_segementation: torch.Tensor | None = None
    denoised_logits: torch.Tensor | None = None
    autoencoded_img: torch.Tensor | None = None

    def __post_init__(self):
        if (
            self.depth_maps is not None
            and self.depth_maps.shape[0] != self.segmentation.shape[0]
        ):
            raise ValueError(
                "depth_maps must have same number of samples as segmentation"
            )
        if (
            self.denoised_segementation is not None
            and self.denoised_segementation.shape[0]
            != self.denoised_segementation.shape[0]
        ):
            raise ValueError(
                "denoised_segementation must have same number of samples as segmentation"
            )
        if self.accuracy.shape[0] != self.segmentation.shape[0]:
            raise ValueError(
                "accuracy must have same number of samples as segmentation"
            )
        if self.logits.shape[0] != self.segmentation.shape[0]:
            raise ValueError("logits must have same number of samples as segmentation")

    def __len__(self):
        return self.segmentation.shape[0]

    def to_device(self, device: torch.device):
        self.segmentation = self.segmentation.to(device)
        self.logits = self.logits.to(device)
        self.accuracy = self.accuracy.to(device)
        if self.depth_maps is not None:
            self.depth_maps = self.depth_maps.to(device)
        if self.denoised_segementation is not None:
            self.denoised_segementation = self.denoised_segementation.to(device)
        if self.denoised_logits is not None:
            self.denoised_logits = self.denoised_logits.to(device)
        if self.autoencoded_img is not None:
            self.autoencoded_img = self.autoencoded_img.to(device)

    def cpu(self):
        self.to_device(torch.device("cpu"))
        return self

    def detach(self):
        self.segmentation = self.segmentation.detach()
        self.logits = self.logits.detach()
        self.accuracy = self.accuracy.detach()
        if self.depth_maps is not None:
            self.depth_maps = self.depth_maps.detach()
        if self.denoised_segementation is not None:
            self.denoised_segementation = self.denoised_segementation.detach()
        if self.denoised_logits is not None:
            self.denoised_logits = self.denoised_logits.detach()
        if self.autoencoded_img is not None:
            self.autoencoded_img = self.autoencoded_img.detach()
        return self

    def __add__(self, other):
        if not isinstance(other, Predictions):
            raise ValueError("other must be Predictions")

        return Predictions(
            segmentation=torch.cat([self.segmentation, other.segmentation], dim=0),
            logits=torch.cat([self.logits, other.logits], dim=0),
            accuracy=torch.cat([self.accuracy, other.accuracy], dim=0),
            depth_maps=(
                torch.cat([self.depth_maps, other.depth_maps], dim=0)
                if self.depth_maps is not None
                else None
            ),
            denoised_segementation=(
                torch.cat(
                    [self.denoised_segementation, other.denoised_segementation], dim=0
                )
                if self.denoised_segementation is not None
                else None
            ),
            denoised_logits=(
                torch.cat([self.denoised_logits, other.denoised_logits], dim=0)
                if self.denoised_logits is not None
                else None
            ),
            autoencoded_img=(
                torch.cat([self.autoencoded_img, other.autoencoded_img], dim=0)
                if self.autoencoded_img is not None
                else None
            ),
        )

    def save(self, folder: str):
        self.to_device(torch.device("cpu"))
        torch.save(self.segmentation, f"{folder}/pred_segmentation.pt")
        torch.save(self.logits, f"{folder}/pred_logits.pt")
        torch.save(self.accuracy, f"{folder}/pred_accuracy.pt")
        if self.depth_maps is not None:
            torch.save(self.depth_maps, f"{folder}/pred_depth_maps.pt")
        if self.denoised_segementation is not None:
            torch.save(
                self.denoised_segementation, f"{folder}/pred_denoised_segmentation.pt"
            )
        if self.denoised_logits is not None:
            torch.save(self.denoised_logits, f"{folder}/pred_denoised_logits.pt")
        if self.autoencoded_img is not None:
            torch.save(self.autoencoded_img, f"{folder}/pred_autoencoded_img.pt")

    @classmethod
    def load(cls, folder: str):
        segmentation = torch.load(f"{folder}/pred_segmentation.pt")
        logits = torch.load(f"{folder}/pred_logits.pt")
        accuracy = torch.load(f"{folder}/pred_accuracy.pt")
        depth_maps = (
            torch.load(f"{folder}/pred_depth_maps.pt")
            if "pred_depth_maps.pt" in os.listdir(folder)
            else None
        )
        denoised_segmentation = (
            torch.load(f"{folder}/pred_denoised_segmentation.pt")
            if "pred_denoised_segmentation.pt" in os.listdir(folder)
            else None
        )
        denoised_logits = (
            torch.load(f"{folder}/pred_denoised_logits.pt")
            if "pred_denoised_logits.pt" in os.listdir(folder)
            else None
        )
        autoencoded_img = (
            torch.load(f"{folder}/pred_autoencoded_img.pt")
            if "pred_autoencoded_img.pt" in os.listdir(folder)
            else None
        )

        return Predictions(
            segmentation=segmentation,
            logits=logits,
            accuracy=accuracy,
            depth_maps=depth_maps,
            denoised_segementation=denoised_segmentation,
            denoised_logits=denoised_logits,
            autoencoded_img=autoencoded_img,
        )

    def to_list(self):
        return [
            Prediction(
                segmentation=self.segmentation[i],
                logits=self.logits[i],
                accuracy=self.accuracy[i].item(),
                depth_map=(self.depth_maps[i] if self.depth_maps is not None else None),
                denoised_segmentation=(
                    self.denoised_segementation[i]
                    if self.denoised_segementation is not None
                    else None
                ),
                denoised_logits=(
                    self.denoised_logits[i]
                    if self.denoised_logits is not None
                    else None
                ),
                autoencoded_img=(
                    self.autoencoded_img[i]
                    if self.autoencoded_img is not None
                    else None
                ),
            )
            for i in range(len(self))
        ]


@dataclass
class DataInput:
    input_image: torch.Tensor
    area_probability: torch.Tensor | None = None
    position: torch.Tensor | None = None


@dataclass
class GroundTruth:
    segmentation: torch.Tensor
    depth_map: torch.Tensor | None = None
    prev_segmentation: torch.Tensor | None = None


@dataclass
class Prediction:
    segmentation: torch.Tensor
    logits: torch.Tensor
    accuracy: float
    depth_map: torch.Tensor | None = None
    denoised_segmentation: torch.Tensor | None = None
    denoised_logits: torch.Tensor | None = None
    autoencoded_img: torch.Tensor | None = None


@dataclass
class PatchDataItems:
    patch_info: PatchInfos
    data_inputs: SegInputs
    ground_truths: SegGroundTruths
    predictions: Predictions | None = None

    def __len__(self):
        return len(self.data_inputs)

    def __getitem__(self, idx):
        return PatchDataItem(
            brain_area=self.patch_info.brain_area[idx],
            section_id=self.patch_info.section_id[idx],
            patch_id=self.patch_info.patch_id[idx],
            fold=self.patch_info.fold[idx],
            is_corner_patch=self.patch_info.is_corner_patch[idx],
            data_input=DataInput(
                input_image=self.data_inputs.input_images[idx],
                area_probability=(
                    self.data_inputs.area_probabilities[idx]
                    if self.data_inputs.area_probabilities is not None
                    else None
                ),
                position=(
                    self.data_inputs.position[idx]
                    if self.data_inputs.position is not None
                    else None
                ),
            ),
            ground_truth=GroundTruth(
                segmentation=self.ground_truths.segmentation[idx],
                depth_map=(
                    self.ground_truths.depth_maps[idx]
                    if self.ground_truths.depth_maps is not None
                    else None
                ),
                prev_segmentation=(
                    self.ground_truths.prev_segmentation[idx]
                    if self.ground_truths.prev_segmentation is not None
                    else None
                ),
            ),
            prediction=(
                Prediction(
                    segmentation=self.predictions.segmentation[idx],
                    logits=self.predictions.logits[idx],
                    accuracy=self.predictions.accuracy[idx].item(),
                    depth_map=(
                        self.predictions.depth_maps[idx]
                        if self.predictions.depth_maps is not None
                        else None
                    ),
                    denoised_segmentation=(
                        self.predictions.denoised_segementation[idx]
                        if self.predictions.denoised_segementation is not None
                        else None
                    ),
                    denoised_logits=(
                        self.predictions.denoised_logits[idx]
                        if self.predictions.denoised_logits is not None
                        else None
                    ),
                    autoencoded_img=(
                        self.predictions.autoencoded_img[idx]
                        if self.predictions.autoencoded_img is not None
                        else None
                    ),
                )
                if self.predictions is not None
                else None
            ),
        )

    def __add__(self, other):
        if not isinstance(other, PatchDataItems):
            raise ValueError("other must be PatchDataItems")

        return PatchDataItems(
            patch_info=self.patch_info + other.patch_info,
            data_inputs=self.data_inputs + other.data_inputs,
            ground_truths=self.ground_truths + other.ground_truths,
            predictions=(
                self.predictions + other.predictions
                if self.predictions is not None
                else None
            ),
        )

    def save(self, folder: str):
        self.patch_info.save(folder)
        self.data_inputs.save(folder)
        self.ground_truths.save(folder)
        if self.predictions is not None:
            self.predictions.save(folder)

    @classmethod
    def load(cls, folder: str):
        patch_info = PatchInfos.load(folder)
        data_inputs = SegInputs.load(folder)
        ground_truths = SegGroundTruths.load(folder)
        predictions = (
            Predictions.load(folder)
            if "pred_segmentation.pt" in os.listdir(folder)
            else None
        )
        return PatchDataItems(
            patch_info=patch_info,
            data_inputs=data_inputs,
            ground_truths=ground_truths,
            predictions=predictions,
        )

    def sample(self, n_data_items):
        if n_data_items > len(self):
            n_data_items = len(self)
        indices = np.random.choice(len(self), n_data_items, replace=False)
        return [self[idx] for idx in indices]

    @property
    def brain_areas(self):
        return set(self.patch_info.brain_area)

    def get_brain_area_example(self, brain_area: str):
        idxs = [
            i for i, ba in enumerate(self.patch_info.brain_area) if ba == brain_area
        ]
        idx = np.random.choice(idxs)
        return self[idx]

    def filter(self, boolean_func: Callable):
        keep_mask = [boolean_func(self[i]) for i in range(len(self))]
        patch_info = PatchInfos(
            brain_area=[
                self.patch_info.brain_area[i] for i in range(len(self)) if keep_mask[i]
            ],
            section_id=[
                self.patch_info.section_id[i] for i in range(len(self)) if keep_mask[i]
            ],
            patch_id=[
                self.patch_info.patch_id[i] for i in range(len(self)) if keep_mask[i]
            ],
            fold=[self.patch_info.fold[i] for i in range(len(self)) if keep_mask[i]],
            is_corner_patch=[
                self.patch_info.is_corner_patch[i]
                for i in range(len(self))
                if keep_mask[i]
            ],
        )
        data_inputs = SegInputs(
            input_images=self.data_inputs.input_images[keep_mask],
            area_probabilities=(
                self.data_inputs.area_probabilities[keep_mask]
                if self.data_inputs.area_probabilities is not None
                else None
            ),
            position=(
                self.data_inputs.position[keep_mask]
                if self.data_inputs.position is not None
                else None
            ),
        )
        ground_truths = SegGroundTruths(
            segmentation=self.ground_truths.segmentation[keep_mask],
            depth_maps=(
                self.ground_truths.depth_maps[keep_mask]
                if self.ground_truths.depth_maps is not None
                else None
            ),
            prev_segmentation=(
                self.ground_truths.prev_segmentation[keep_mask]
                if self.ground_truths.prev_segmentation is not None
                else None
            ),
        )
        predictions = (
            Predictions(
                segmentation=self.predictions.segmentation[keep_mask],
                logits=self.predictions.logits[keep_mask],
                accuracy=self.predictions.accuracy[keep_mask],
                depth_maps=(
                    self.predictions.depth_maps[keep_mask]
                    if self.predictions.depth_maps is not None
                    else None
                ),
                denoised_segementation=(
                    self.predictions.denoised_segementation[keep_mask]
                    if self.predictions.denoised_segementation is not None
                    else None
                ),
                denoised_logits=(
                    self.predictions.denoised_logits[keep_mask]
                    if self.predictions.denoised_logits is not None
                    else None
                ),
                autoencoded_img=(
                    self.predictions.autoencoded_img[keep_mask]
                    if self.predictions.autoencoded_img is not None
                    else None
                ),
            )
            if self.predictions is not None
            else None
        )
        return PatchDataItems(
            patch_info=patch_info,
            data_inputs=data_inputs,
            ground_truths=ground_truths,
            predictions=predictions,
        )

    def detach(self):
        self.data_inputs.detach()
        self.ground_truths.detach()
        if self.predictions is not None:
            self.predictions.detach()

        return self

    def cpu(self):
        self.data_inputs.cpu()
        self.ground_truths.cpu()
        if self.predictions is not None:
            self.predictions.cpu()

        return self


@dataclass
class PatchDataItem:
    brain_area: str
    section_id: int
    patch_id: int
    fold: int
    is_corner_patch: bool
    data_input: DataInput
    ground_truth: GroundTruth
    prediction: Prediction | None = None

    @property
    def input_image(self):
        return (
            cort.manip.unpad_img(
                self.data_input.input_image, self.ground_truth.segmentation
            )
            .squeeze()
            .numpy()
        )

    @property
    def gt_segmentation(self):
        return cort.manip.unpad_mask(self.ground_truth.segmentation).squeeze().numpy()

    @property
    def pred_segmentation(self):
        return (
            cort.manip.unpad_img(
                self.prediction.segmentation, self.ground_truth.segmentation
            )
            .squeeze()
            .numpy()
        )

    @property
    def pred_denoised_segmentation(self):
        if self.prediction.denoised_segmentation is None:
            return None
        return (
            cort.manip.unpad_img(
                self.prediction.denoised_segmentation, self.ground_truth.segmentation
            )
            .squeeze()
            .numpy()
        )

    @property
    def existing_cort_layers(self):
        if self.ground_truth.prev_segmentation is None:
            return None
        return (
            cort.manip.unpad_img(
                self.ground_truth.prev_segmentation, self.ground_truth.segmentation
            )
            .squeeze()
            .numpy()
        )

    @property
    def autoencoded_img(self):
        if (
            self.prediction.autoencoded_img is None
            or self.ground_truth.segmentation is None
        ):
            return None
        return (
            cort.manip.unpad_img(
                self.prediction.autoencoded_img, self.ground_truth.segmentation
            )
            .squeeze()
            .numpy()
        )

    @property
    def gt_depth_map(self):
        if (
            self.ground_truth.depth_map is None
            or self.ground_truth.segmentation is None
        ):
            return None
        return (
            cort.manip.unpad_img(
                self.ground_truth.depth_map, self.ground_truth.segmentation
            )
            .squeeze()
            .numpy()
        )

    @property
    def pred_depth_map(self):
        if self.prediction.depth_map is None or self.ground_truth.segmentation is None:
            return None
        return (
            cort.manip.unpad_img(
                self.prediction.depth_map, self.ground_truth.segmentation
            )
            .squeeze()
            .numpy()
        )
