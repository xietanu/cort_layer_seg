import json
import os

import numpy as np
import torch

import datasets
import nnet.protocols
import nnet.models
import nnet.training


TEMP_MODEL = "models/unet_temp.pth"


def predict_from_model(
    fold: int,
    fold_data: datasets.Fold,
    outputs_path: str,
    model: nnet.protocols.ModelProtocol | None = None,
    train_log: nnet.training.TrainLog | None = None,
):
    if model is None:
        model = nnet.models.SemantSegUNetModel.restore(TEMP_MODEL)

    print("Saving training logs...")
    os.makedirs((os.path.join(outputs_path, "logs")), exist_ok=True)
    json.dump(
        train_log.train_losses,
        open(os.path.join(outputs_path, "logs", "train_losses.json"), "w"),
    )
    json.dump(
        train_log.val_losses,
        open(os.path.join(outputs_path, "logs", "val_losses.json"), "w"),
    )
    json.dump(
        train_log.train_accs,
        open(os.path.join(outputs_path, "logs", "train_accs.json"), "w"),
    )
    json.dump(
        train_log.val_accs,
        open(os.path.join(outputs_path, "logs", "val_accs.json"), "w"),
    )

    print("Predicting test set...")
    base_results = get_seg_output_for_loader(model, fold_data.test_dataloader)
    print("Saving test set results...")
    os.makedirs(os.path.join(outputs_path, "test"), exist_ok=True)
    base_results.save(os.path.join(outputs_path, "test"))

    # print("Predicting transposed test set...")
    # transposed_results = get_seg_output_for_loader(
    #    model, fold_data.test_dataloader, transpose=True
    # )
    # print("Saving transposed test set results...")
    # os.makedirs(os.path.join(outputs_path, "test_transposed"), exist_ok=True)
    # transposed_results.save(os.path.join(outputs_path, "test_transposed"))

    print("Predicting on Siibra patches test set...")
    siibra_results = get_seg_output_for_loader(model, fold_data.siibra_test_dataloader)
    print("Saving Siibra patches test set results...")
    os.makedirs(os.path.join(outputs_path, "siibra_test"), exist_ok=True)
    siibra_results.save(os.path.join(outputs_path, "siibra_test"))

    print("Done!")


def get_seg_output_for_loader(
    model: nnet.protocols.ModelProtocol,
    loader: torch.utils.data.DataLoader,
    transpose: bool = False,
) -> datasets.datatypes.PatchDataItems:
    """Get the segmentation output for a data loader."""
    seg_output = None

    for batch in loader:
        batch_info, batch_inputs, batch_gt = batch
        batch_info = datasets.datatypes.PatchInfos(
            brain_area=batch_info[0],
            section_id=batch_info[1].detach().cpu().numpy().tolist(),
            patch_id=batch_info[2].detach().cpu().numpy().tolist(),
            fold=batch_info[3].detach().cpu().numpy().tolist(),
        )
        batch_inputs = datasets.datatypes.DataInputs(*batch_inputs)
        batch_gt = datasets.datatypes.GroundTruths(*batch_gt)
        if transpose:
            batch_inputs.input_images = batch_inputs.input_images.transpose(2, 3)

        batch_output = model.predict(batch_inputs)

        batch_data_items = datasets.datatypes.PatchDataItems(
            patch_info=batch_info,
            data_inputs=batch_inputs,
            ground_truths=batch_gt,
            predictions=batch_output,
        )

        if seg_output is None:
            seg_output = batch_data_items
        else:
            seg_output += batch_data_items

    return seg_output
