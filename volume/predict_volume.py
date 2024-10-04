import numpy as np
import torch
from tqdm import tqdm

import datasets
import nnet.protocols


def predict_logits(
    model: nnet.protocols.SegModelProtocol,
    volume: np.ndarray,
    position: np.ndarray,
    probabilities: np.ndarray,
    denoised: bool = True,
) -> torch.Tensor:
    padded_images = [
        np.pad(img, ((0, 256 - img.shape[0]), (0, 256 - img.shape[1])))
        for img in volume
    ]

    predictions = None

    for i in tqdm(range(0, len(padded_images), 4)):
        cur_imgs = torch.tensor(np.stack(padded_images[i : i + 4], axis=0)).unsqueeze(1)
        input_data_batch = datasets.datatypes.SegInputs(
            input_images=cur_imgs,
            position=torch.tensor(position).repeat(len(cur_imgs), 1),
            area_probabilities=torch.tensor(probabilities).repeat(len(cur_imgs), 1),
        )
        cur_pred = model.predict(input_data_batch)
        if predictions is None:
            predictions = cur_pred
        else:
            predictions += cur_pred

    if denoised:
        denoised_logits = predictions.denoised_logits[
            :, :, : volume.shape[1], : volume.shape[2]
        ]
    else:
        denoised_logits = predictions.logits[:, :, : volume.shape[1], : volume.shape[2]]

    return denoised_logits.permute(1, 0, 2, 3)


def predict_segmentation(
    model: nnet.protocols.SegModelProtocol,
    volume: np.ndarray,
    position: np.ndarray,
    probabilities: np.ndarray,
    perp_smooth: bool = True,
    denoised: bool = True,
) -> np.ndarray:
    padded_images = [
        np.pad(img, ((0, 256 - img.shape[0]), (0, 256 - img.shape[1])))
        for img in volume
    ]

    predictions = None

    for i in tqdm(range(0, len(padded_images), 4)):
        cur_imgs = torch.tensor(np.stack(padded_images[i : i + 4], axis=0)).unsqueeze(1)
        input_data_batch = datasets.datatypes.SegInputs(
            input_images=cur_imgs,
            position=torch.tensor(position).repeat(len(cur_imgs), 1),
            area_probabilities=torch.tensor(probabilities).repeat(len(cur_imgs), 1),
        )
        cur_pred = model.predict(input_data_batch)
        if predictions is None:
            predictions = cur_pred
        else:
            predictions += cur_pred

    if denoised:
        denoised_logits = predictions.denoised_logits[
            :, :, : volume.shape[1], : volume.shape[2]
        ]
    else:
        denoised_logits = predictions.logits[:, :, : volume.shape[1], : volume.shape[2]]

    if not perp_smooth:
        return denoised_logits.argmax(axis=1).detach().cpu().numpy()

    denoised_logits = denoised_logits.permute(3, 1, 2, 0).detach().cpu().numpy()

    orig_shape = denoised_logits.shape

    padded_images = [
        np.pad(img, ((0, 0), (0, 256 - img.shape[1]), (0, 128 - len(padded_images))))
        for img in denoised_logits
    ]

    side_predictions = []

    for i in tqdm(range(0, len(padded_images), 4)):
        cur_images = np.array(padded_images[i : i + 4])
        input_data = datasets.datatypes.SegInputs(
            input_images=torch.tensor(cur_images),
            position=torch.tensor(position).repeat(len(cur_images), 1),
            area_probabilities=torch.tensor(probabilities).repeat(len(cur_images), 1),
        )
        _, side_pred = model.denoise_model.predict(
            input_data.input_images, input_data.area_probabilities, input_data.position
        )
        side_predictions.append(side_pred)

    side_predictions_stacked = (
        torch.concat(side_predictions, dim=0)
        .squeeze()[: orig_shape[0], : orig_shape[2], : orig_shape[3]]
        .permute(2, 1, 0)
        .cpu()
        .detach()
        .numpy()
    )

    return side_predictions_stacked


def denoise_segmentation(
    model: nnet.protocols.SegModelProtocol,
    volume: np.ndarray,
    position: np.ndarray,
    probabilities: np.ndarray,
):
    orig_shape = volume.shape

    padded_volume = np.pad(
        volume, ((0, 0), (0, 0), (0, 256 - volume.shape[2]), (0, 256 - volume.shape[3]))
    )

    predictions = []

    for i in tqdm(range(0, padded_volume.shape[1], 4)):
        cur_images = np.array(padded_volume[:, i : i + 4]).transpose(1, 0, 2, 3)
        input_data = datasets.datatypes.SegInputs(
            input_images=torch.tensor(cur_images),
            position=torch.tensor(position).repeat(len(cur_images), 1),
            area_probabilities=torch.tensor(probabilities).repeat(len(cur_images), 1),
        )
        side_pred, _ = model.denoise_model.predict(
            input_data.input_images, input_data.area_probabilities, input_data.position
        )
        predictions.append(side_pred)

    predictions_stacked = (
        torch.concat(predictions, dim=0)
        .squeeze()[:, : orig_shape[0], : orig_shape[2], : orig_shape[3]]
        .permute(1, 0, 2, 3)
        .cpu()
        .detach()
        .numpy()
    )

    return predictions_stacked
