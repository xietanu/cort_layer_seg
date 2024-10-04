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
        use_transforms=False,
        use_prev_seg_with_siibra=False,
        padded_size=(256, 128),
        use_synth=True,
    )
    siibra_fold_data = datasets.load_fold(
        fold=fold,
        condition=condition,
        use_position=config["positional"],
        perc_siibra=1.0,
        roll=0.75,
        use_transforms=False,
        use_prev_seg_with_siibra=True,
        padded_size=(256, 128),
    )

    model = nnet.models.SemantSegUNetModel.restore(TEMP_MODEL)

    print("Predicting train set on base...")
    data = get_seg_output_for_loader(model, fold_data.train_dataloader, max_batches=200)
    print(f"Predicted {len(data)} patches.")
    # data = data.filter(filter_by_f1_score)
    # print(f"Filtered to {len(data)} patches.")
    print("Predicting train set on siibra...")
    siibra_data = get_seg_output_for_loader(model, siibra_fold_data.train_dataloader)
    print(f"Predicted {len(siibra_data)} patches.")
    # siibra_data = siibra_data.filter(filter_by_f1_score)
    # print(f"Filtered to {len(siibra_data)} patches.")
    # data = siibra_data
    data += siibra_data
    print("Predicting train set on synthetic...")
    syn_data = get_seg_output_for_loader(model, fold_data.synth_dataloader)
    print(f"Predicted {len(syn_data)} patches.")
    # syn_data = syn_data.filter(filter_by_f1_score)
    # print(f"Filtered to {len(syn_data)} patches.")
    data += syn_data

    print("Saving train set results...")
    os.makedirs("data/denoise_data", exist_ok=True)
    data.save("data/denoise_data")
    print("Predicting val set...")
    data = get_seg_output_for_loader(model, fold_data.val_dataloader)
    siibra_data = get_seg_output_for_loader(model, siibra_fold_data.val_dataloader)
    data += siibra_data
    # data = siibra_data
    print(f"Predicted {len(data)} patches.")
    # data = data.filter(filter_by_f1_score)
    # print(f"Filtered to {len(data)} patches.")
    print("Saving val set results...")
    os.makedirs("data/denoise_data_val", exist_ok=True)
    data.save("data/denoise_data_val")
    print("Done!")


def filter_by_f1_score(result: datasets.datatypes.PatchDataItem) -> bool:
    f1_score = evaluate.f1_score(
        result.prediction.segmentation,
        result.ground_truth.segmentation,
        n_classes=8,
        ignore_index=cort.constants.PADDING_MASK_VALUE,
    )
    return f1_score >= 0.75


def get_seg_output_for_loader(
    model: nnet.protocols.SegModelProtocol,
    loader: torch.utils.data.DataLoader,
    transpose: bool = False,
    max_batches: int = 96,
) -> datasets.datatypes.PatchDataItems:
    """Get the segmentation output for a data loader."""
    seg_outputs = []

    n_batches = 0

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

        with torch.no_grad():
            batch_output = model.predict(batch_inputs)

        batch_data_items = (
            datasets.datatypes.PatchDataItems(
                patch_info=batch_info,
                data_inputs=batch_inputs,
                ground_truths=batch_gt,
                predictions=batch_output,
            )
            .detach()
            .cpu()
        )

        seg_outputs.append(batch_data_items)

        n_batches += 1
        if n_batches >= max_batches:
            break

    # Though more complicated, merging outputs pairwise recursively is much faster
    # 172 batches merging goes from ~60s if done naively to ~2s this way
    # Savings much greater for larger numbers of batches
    merged_seg_outputs = seg_outputs
    merged_count = 0
    while len(merged_seg_outputs) > 1:
        seg_outputs = merged_seg_outputs
        merged_seg_outputs = []
        print(f"{merged_count+1}. Merging {len(seg_outputs)} outputs...")
        for i in tqdm(range(0, len(seg_outputs), 2), desc="Merging outputs"):
            if i + 1 < len(seg_outputs):
                merged_seg_outputs.append(seg_outputs[i] + seg_outputs[i + 1])
            else:
                merged_seg_outputs.append(seg_outputs[i])
        merged_count += 1

    seg_output = merged_seg_outputs[0]

    return seg_output


if __name__ == "__main__":
    main()
