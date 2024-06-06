import os
from dataclasses import dataclass

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

    def __add__(self, other):
        if not isinstance(other, PatchInfos):
            raise ValueError("other must be PatchInfos")

        return PatchInfos(
            brain_area=self.brain_area + other.brain_area,
            section_id=self.section_id + other.section_id,
            patch_id=self.patch_id + other.patch_id,
            fold=self.fold + other.fold,
        )

    def save(self, folder: str):
        with open(f"{folder}/patch_infos.json", "w") as f:
            json.dump(
                {
                    "brain_area": self.brain_area,
                    "section_id": self.section_id,
                    "patch_id": self.patch_id,
                    "fold": self.fold,
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
        )


@dataclass
class DataInputs:
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

    def __add__(self, other):
        if not isinstance(other, DataInputs):
            raise ValueError("other must be DataInputs")

        return DataInputs(
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
        return DataInputs(
            input_images=input_images,
            area_probabilities=area_probabilities,
            position=position,
        )


@dataclass
class GroundTruths:
    segmentation: torch.Tensor
    depth_maps: torch.Tensor | None = None

    def __post_init__(self):
        if (
            self.depth_maps is not None
            and self.depth_maps.shape[0] != self.segmentation.shape[0]
        ):
            raise ValueError(
                "depth_maps must have same number of samples as segmentation"
            )

    def __len__(self):
        return self.segmentation.shape[0]

    def to_device(self, device: torch.device):
        self.segmentation = self.segmentation.to(device)
        if self.depth_maps is not None:
            self.depth_maps = self.depth_maps.to(device)

    def __add__(self, other):
        if not isinstance(other, GroundTruths):
            raise ValueError("other must be GroundTruths")

        return GroundTruths(
            segmentation=torch.cat([self.segmentation, other.segmentation], dim=0),
            depth_maps=(
                torch.cat([self.depth_maps, other.depth_maps], dim=0)
                if self.depth_maps is not None
                else None
            ),
        )

    def save(self, folder: str):
        self.to_device(torch.device("cpu"))
        torch.save(self.segmentation, f"{folder}/gt_segmentation.pt")
        if self.depth_maps is not None:
            torch.save(self.depth_maps, f"{folder}/gt_depth_maps.pt")

    @classmethod
    def load(cls, folder: str):
        segmentation = torch.load(f"{folder}/gt_segmentation.pt")
        depth_maps = (
            torch.load(f"{folder}/gt_depth_maps.pt")
            if "gt_depth_maps.pt" in os.listdir(folder)
            else None
        )
        return GroundTruths(segmentation=segmentation, depth_maps=depth_maps)


@dataclass
class Predictions:
    segmentation: torch.Tensor
    logits: torch.Tensor
    accuracy: torch.Tensor
    depth_maps: torch.Tensor | None = None

    def __post_init__(self):
        if (
            self.depth_maps is not None
            and self.depth_maps.shape[0] != self.segmentation.shape[0]
        ):
            raise ValueError(
                "depth_maps must have same number of samples as segmentation"
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
        )

    def save(self, folder: str):
        self.to_device(torch.device("cpu"))
        torch.save(self.segmentation, f"{folder}/pred_segmentation.pt")
        torch.save(self.logits, f"{folder}/pred_logits.pt")
        torch.save(self.accuracy, f"{folder}/pred_accuracy.pt")
        if self.depth_maps is not None:
            torch.save(self.depth_maps, f"{folder}/pred_depth_maps.pt")

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
        return Predictions(
            segmentation=segmentation,
            logits=logits,
            accuracy=accuracy,
            depth_maps=depth_maps,
        )

    def to_list(self):
        return [
            Prediction(
                segmentation=self.segmentation[i],
                logits=self.logits[i],
                accuracy=self.accuracy[i].item(),
                depth_map=(self.depth_maps[i] if self.depth_maps is not None else None),
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


@dataclass
class Prediction:
    segmentation: torch.Tensor
    logits: torch.Tensor
    accuracy: float
    depth_map: torch.Tensor | None = None


@dataclass
class PatchDataItems:
    patch_info: PatchInfos
    data_inputs: DataInputs
    ground_truths: GroundTruths
    predictions: Predictions | None = None

    def __len__(self):
        return len(self.data_inputs)

    def __getitem__(self, idx):
        return PatchDataItem(
            brain_area=self.patch_info.brain_area[idx],
            section_id=self.patch_info.section_id[idx],
            patch_id=self.patch_info.patch_id[idx],
            fold=self.patch_info.fold[idx],
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
        data_inputs = DataInputs.load(folder)
        ground_truths = GroundTruths.load(folder)
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


@dataclass
class PatchDataItem:
    brain_area: str
    section_id: int
    patch_id: int
    fold: int
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
