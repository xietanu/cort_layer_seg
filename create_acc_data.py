import argparse
import json
import os

import torch
from tqdm import tqdm

import datasets
import evaluate
import datasets.datatypes
import cort
import nnet.models
import nnet.protocols

CONFIGS_PATH = "experiments/configs"
TEMP_MODEL = "models/unet_temp.pth"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_name", type=str)
    parser.add_argument("fold", type=int)

    args = parser.parse_args()

    config_name = args.config_name
    fold = args.fold

    print("Loading model...")

    config = json.load(open(f"{CONFIGS_PATH}/{config_name}.json"))

    condition = None
    if config["conditional"]:
        condition = datasets.dataset_conditions_lookup[
            config["dataset_config"]["condition"]
        ]

    fold_data = datasets.load_fold(
        fold=fold,
        condition=condition,
        use_position=config["positional"],
        perc_siibra=0.0,
        roll=0.75,
        use_transforms=True,
        use_prev_seg_with_siibra=False,
        padded_size=(256, 128),
    )
    siibra_fold_data = datasets.load_fold(
        fold=fold,
        condition=condition,
        use_position=config["positional"],
        perc_siibra=1.0,
        roll=0.75,
        use_transforms=False,
        use_prev_seg_with_siibra=False,
        padded_size=(256, 128),
    )

    model = nnet.models.SemantSegUNetModel.restore(TEMP_MODEL)

    print("Predicting train set on base...")
    data = get_seg_output_for_loader(model, fold_data.train_dataloader)
    print(f"Predicted {len(data)} patches.")
    print("Predicting train set on siibra...")
    siibra_data = get_seg_output_for_loader(model, siibra_fold_data.train_dataloader)
    print(f"Filtered to {len(siibra_data)} patches.")
    # data = siibra_data
    data += siibra_data
    print("Saving train set results...")
    os.makedirs("data/acc_data", exist_ok=True)
    data.save("data/acc_data")
    print("Predicting val set...")
    data = get_seg_output_for_loader(model, fold_data.val_dataloader)
    siibra_data = get_seg_output_for_loader(model, siibra_fold_data.val_dataloader)
    data += siibra_data
    # data = siibra_data
    print(f"Predicted {len(data)} patches.")
    print("Saving val set results...")
    os.makedirs("data/acc_data_val", exist_ok=True)
    data.save("data/acc_data_val")
    print("Done!")


def get_seg_output_for_loader(
    model: nnet.protocols.SegModelProtocol,
    loader: torch.utils.data.DataLoader,
    transpose: bool = False,
) -> datasets.datatypes.PatchDataItems:
    """Get the segmentation output for a data loader."""
    seg_output = None

    for batch in tqdm(loader, desc="Predicting batches"):
        batch_info, batch_inputs, batch_gt = batch
        batch_info = datasets.datatypes.PatchInfos(
            brain_area=batch_info[0],
            section_id=batch_info[1].detach().cpu().numpy().tolist(),
            patch_id=batch_info[2].detach().cpu().numpy().tolist(),
            fold=batch_info[3].detach().cpu().numpy().tolist(),
            is_corner_patch=batch_info[4].detach().cpu().numpy().tolist(),
        )
        batch_inputs = datasets.datatypes.SegInputs(*batch_inputs)
        batch_gt = datasets.datatypes.SegGroundTruths(*batch_gt)
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


if __name__ == "__main__":
    main()
