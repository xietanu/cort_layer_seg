import ctypes
import json
import os
from io import BytesIO

import PySimpleGUI as sg
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

import cort
import cort.display
import datasets
import evaluate
import experiments
import nnet.training

EXP_BASE_PATH = "experiments/results"

IMAGE_SIZE = (1500, 900)

MACRO_F1 = "macro_f1"
MACRO_F1_DENOISED = "macro_f1_denoised"
F1_BY_CLASS = "f1_by_class"
F1_BY_CLASS_DENOISED = "f1_by_class_denoised"
F1_BY_AREA = "f1_by_area"
F1_BY_AREA_DENOISED = "f1_by_area_denoised"
F1_BEST_CUTOFF = "f1_best_cutoff"
F1_WORST_CUTOFF = "f1_worst_cutoff"
PPA = "per_pixel_accuracy"
PPA_DENOISED = "per_pixel_accuracy_denoised"


ALL = "all"
MAIN_ONLY = "main_only"
CORNER_ONLY = "corner_only"
BEST = "best"
WORST = "worst"

ORIGINAL = "-ORIGINAL-"
SIIBRA = "-SIIBRA-"

origin_fp = {
    ORIGINAL: "test",
    SIIBRA: "siibra_test",
}

INPUT_IMAGE = "-INPUT_IMAGE-"
GT_SEGMENTATION = "-GT_SEGMENTATION-"
PRED_SEGMENTATION = "-PRED_SEGMENTATION-"
PRED_SEGMENTATION_SHOW_ERRORS = "-PRED_SEGMENTATION_SHOW_ERRORS-"
PRED_DENOISED_SEGMENTATION = "-PRED_DENOISED_SEGMENTATION-"
PREV_DENOISED_SEGMENTATION_SHOW_ERRORS = "-PREV_DENOISED_SEGMENTATION_SHOW_ERRORS-"
PRED_DEPTH_MAP = "-PRED_DEPTH_MAP-"
AUTOENCODED_IMG = "-AUTOENCODED_IMG-"


def main():
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
    cur_experiment = None
    all_results = None
    cur_results = None
    cur_patch_type = ALL
    cur_areas = ALL
    n_patches = 5
    cur_origin = ORIGINAL

    to_display = {
        INPUT_IMAGE: False,
        GT_SEGMENTATION: False,
        PRED_SEGMENTATION: True,
        PRED_SEGMENTATION_SHOW_ERRORS: False,
        PRED_DENOISED_SEGMENTATION: False,
        PREV_DENOISED_SEGMENTATION_SHOW_ERRORS: True,
        PRED_DEPTH_MAP: False,
        AUTOENCODED_IMG: False,
    }

    seed = 0

    cached_stats = {
        ORIGINAL: {},
        SIIBRA: {},
    }

    experiment_folders = os.listdir("experiments/results")
    experiment_folders = [
        folder
        for folder in experiment_folders
        if os.path.isdir(os.path.join(EXP_BASE_PATH, folder))
    ]

    sidebar = [
        [
            sg.Text("Experiment:"),
            sg.DropDown(experiment_folders, key="-EXPERIMENT-", enable_events=True),
        ],
        [sg.HSeparator()],
        [
            sg.Text("Seed:"),
            sg.In(key="-SEED-", size=(5, 1), default_text="0", enable_events=True),
        ],
        [sg.HSeparator()],
        [sg.Text("Origin:")],
        [
            sg.Radio(
                "HQ patches",
                "origin_type",
                key=ORIGINAL,
                default=True,
                enable_events=True,
            ),
            sg.Radio("Siibra", "origin_type", key=SIIBRA, enable_events=True),
        ],
        [sg.Text("Patch type:")],
        [
            sg.Radio(
                "All", "patch_type", key="-ALL-", default=True, enable_events=True
            ),
            sg.Radio("Main Only", "patch_type", key="-MAIN_ONLY-", enable_events=True),
            sg.Radio(
                "Corner Only", "patch_type", key="-CORNER_ONLY-", enable_events=True
            ),
        ],
        [sg.HSeparator()],
        [sg.Text("Areas:")],
        [
            sg.Radio(
                "All", "areas", key="-ALL_AREAS-", default=True, enable_events=True
            ),
            sg.Radio("Best", "areas", key="-BEST_AREAS-", enable_events=True),
            sg.Radio("Worst", "areas", key="-WORST_AREAS-", enable_events=True),
        ],
        [sg.HSeparator()],
        [sg.Text("Display:")],
        [
            sg.Checkbox(
                "Input Image",
                key=INPUT_IMAGE,
                default=False,
                enable_events=True,
            ),
        ],
        [
            sg.Checkbox(
                "GT Segmentation",
                key=GT_SEGMENTATION,
                default=False,
                enable_events=True,
            ),
        ],
        [
            sg.Checkbox(
                "Pred Segmentation",
                key=PRED_SEGMENTATION,
                default=True,
                enable_events=True,
            ),
        ],
        [
            sg.Checkbox(
                "Pred Segmentation (show errors)",
                key=PRED_SEGMENTATION_SHOW_ERRORS,
                default=False,
                enable_events=True,
            ),
        ],
        [
            sg.Checkbox(
                "Pred Denoised Segmentation",
                key=PRED_DENOISED_SEGMENTATION,
                default=False,
                enable_events=True,
            ),
        ],
        [
            sg.Checkbox(
                "Prev Denoised Segmentation (show errors)",
                key=PREV_DENOISED_SEGMENTATION_SHOW_ERRORS,
                default=True,
                enable_events=True,
            ),
        ],
        [
            sg.Checkbox(
                "Pred Depth Map",
                key=PRED_DEPTH_MAP,
                default=False,
                enable_events=True,
            ),
        ],
        [
            sg.Checkbox(
                "Autoencoded Image",
                key=AUTOENCODED_IMG,
                default=False,
                enable_events=True,
            ),
        ],
        [sg.HSeparator()],
        [sg.Text("# patches:")],
        [
            sg.Slider(
                range=(1, 20),
                default_value=5,
                orientation="horizontal",
                key="-N_PATCHES-",
                enable_events=True,
            )
        ],
    ]

    main_viewer = [
        [
            sg.Image(key="-IMAGE-", size=IMAGE_SIZE),
        ],
    ]

    training_plot = [
        [
            sg.Image(key="-TRAIN_PLOT-", size=IMAGE_SIZE),
        ],
    ]

    stats_page = [
        [sg.Button("Calculate Stats", key="-CALC_STATS-", enable_events=True)],
        [sg.HSeparator()],
        [sg.Text(f"Macro F1: -", key="-MACRO_F1-")],
        [
            sg.Text(
                f"Macro F1 Denoised: -",
                key="-MACRO_F1_DENOISED-",
            )
        ],
        [sg.HSeparator()],
        [sg.Text(f"Per pixel accuracy: -", key="-PPA-")],
        [
            sg.Text(
                f"Per pixel accuracy (denoised): -",
                key="-PPA_DENOISED-",
            )
        ],
        [sg.HSeparator()],
        [
            sg.Text(
                "Macro F1 by class (F1 / Denoised F1):\n-", key="-MACRO_F1_BY_CLASS-"
            )
        ],
        [sg.HSeparator()],
        [sg.Text("Macro F1 by area (F1 / Denoised F1):\n-", key="-MACRO_F1_BY_AREA-")],
    ]

    tab_group = sg.TabGroup(
        [
            [sg.Tab("Main Viewer", main_viewer)],
            [sg.Tab("Stats", stats_page)],
            [sg.Tab("Training", training_plot)],
        ]
    )

    layout = [
        [
            sg.Column(sidebar),
            sg.VSeparator(),
            sg.Column(
                [[tab_group]],
                scrollable=True,
                vertical_scroll_only=True,
                size=IMAGE_SIZE,
                key="-VIEWER-",
            ),
        ],
    ]

    # Create the window
    window = sg.Window("Patch Viewer", layout, resizable=True, finalize=True)

    window["-IMAGE-"].update(
        data=array_to_data(np.zeros((IMAGE_SIZE[1], IMAGE_SIZE[0], 4), dtype=np.uint8))
    )

    # Create an event loop
    while True:
        event, values = window.read()
        # End program if user closes window or
        # presses the OK button
        if event == "OK" or event == sg.WIN_CLOSED:
            break

        if event == ORIGINAL or event == SIIBRA:
            cur_origin = event

            all_results = datasets.datatypes.PatchDataItems.load(
                f"experiments/results/{cur_experiment}/{origin_fp[cur_origin]}"
            )
            cur_results = filter_results(
                all_results,
                cur_experiment,
                cur_origin,
                cur_patch_type,
                cur_areas,
                cached_stats,
            )
            update(
                cur_results,
                window,
                to_display,
                cur_experiment,
                cur_origin,
                cur_patch_type,
                seed,
                cached_stats,
                n_patches,
            )

        if event == "-EXPERIMENT-":
            new_experiment = values["-EXPERIMENT-"]
            if cur_experiment != new_experiment:
                cur_experiment = new_experiment

                if os.path.exists(
                    f"experiments/results/{cur_experiment}/train_log_0.json"
                ):
                    train_log = nnet.training.Log.load(
                        f"experiments/results/{cur_experiment}/train_log_0.json"
                    )
                    update_train_plot(train_log, window)

                all_results = datasets.datatypes.PatchDataItems.load(
                    f"experiments/results/{cur_experiment}/{origin_fp[cur_origin]}"
                )
                cur_results = filter_results(
                    all_results,
                    cur_experiment,
                    cur_origin,
                    cur_patch_type,
                    cur_areas,
                    cached_stats,
                )
                update(
                    cur_results,
                    window,
                    to_display,
                    cur_experiment,
                    cur_origin,
                    cur_patch_type,
                    seed,
                    cached_stats,
                    n_patches,
                )

        if event == "-SEED-":
            if np.all([ch.isnumeric() for ch in values["-SEED-"]]):
                new_seed = int(values["-SEED-"])
                if new_seed != seed and cur_results is not None:
                    seed = new_seed
                    update(
                        cur_results,
                        window,
                        to_display,
                        cur_experiment,
                        cur_origin,
                        cur_patch_type,
                        seed,
                        cached_stats,
                        n_patches,
                    )

        if event in ["-ALL-", "-MAIN_ONLY-", "-CORNER_ONLY-"]:
            if event == "-ALL-":
                cur_patch_type = ALL
            elif event == "-MAIN_ONLY-":
                cur_patch_type = MAIN_ONLY
            elif event == "-CORNER_ONLY-":
                cur_patch_type = CORNER_ONLY
            else:
                raise ValueError(f"Unknown event: {event}")
            if all_results is not None:
                cur_results = filter_results(
                    all_results,
                    cur_experiment,
                    cur_origin,
                    cur_patch_type,
                    cur_areas,
                    cached_stats,
                )
                update(
                    cur_results,
                    window,
                    to_display,
                    cur_experiment,
                    cur_origin,
                    cur_patch_type,
                    seed,
                    cached_stats,
                    n_patches,
                )

        if event == "-CALC_STATS-":
            if (
                cur_experiment not in cached_stats
                or cur_patch_type not in cached_stats[cur_experiment]
            ):
                write_stats(window, None)
                continue

            if cached_stats[cur_experiment][cur_patch_type] is None:
                stats = calc_stats(cur_results)
                cached_stats[cur_experiment][cur_patch_type] = stats

            stats = cached_stats[cur_experiment][cur_patch_type]
            write_stats(window, stats)

        if event in ["-ALL_AREAS-", "-BEST_AREAS-", "-WORST_AREAS-"]:
            if event == "-ALL_AREAS-":
                cur_areas = ALL
            elif event == "-BEST_AREAS-":
                cur_areas = BEST
            elif event == "-WORST_AREAS-":
                cur_areas = WORST
            else:
                raise ValueError(f"Unknown event: {event}")

            if all_results is not None:
                cur_results = filter_results(
                    all_results,
                    cur_experiment,
                    cur_origin,
                    cur_patch_type,
                    cur_areas,
                    cached_stats,
                )
                update(
                    cur_results,
                    window,
                    to_display,
                    cur_experiment,
                    cur_origin,
                    cur_patch_type,
                    seed,
                    cached_stats,
                    n_patches,
                )

        for key in to_display:
            if event == key:
                to_display[key] = values[key]
                if cur_results is not None:
                    update(
                        cur_results,
                        window,
                        to_display,
                        cur_experiment,
                        cur_origin,
                        cur_patch_type,
                        seed,
                        cached_stats,
                        n_patches,
                    )

        if event == "-N_PATCHES-":
            n_patches = int(values["-N_PATCHES-"])
            if cur_results is not None:
                update(
                    cur_results,
                    window,
                    to_display,
                    cur_experiment,
                    cur_origin,
                    cur_patch_type,
                    seed,
                    cached_stats,
                    n_patches,
                )

    window.close()


def update(
    results,
    window,
    to_display,
    cur_experiment,
    cur_origin,
    cur_patch_type,
    seed,
    cached_stats,
    n_patches,
):
    if cur_experiment not in cached_stats[cur_origin]:
        cached_stats[cur_origin][cur_experiment] = {
            cur_patch_type: None,
        }
    elif cur_patch_type not in cached_stats[cur_origin][cur_experiment]:
        cached_stats[cur_origin][cur_experiment][cur_patch_type] = None
    if cached_stats[cur_origin][cur_experiment][
        cur_patch_type
    ] == None and os.path.exists(
        f"experiments/results/{cur_experiment}/{origin_fp[cur_origin]}/{cur_patch_type}_stats.json"
    ):
        cached_stats[cur_origin][cur_experiment][cur_patch_type] = json.load(
            open(
                f"experiments/results/{cur_experiment}/{origin_fp[cur_origin]}/{cur_patch_type}_stats.json"
            )
        )

    stats = cached_stats[cur_origin][cur_experiment][cur_patch_type]

    write_stats(window, stats)

    img = draw_plot(results, seed, to_display, n_patches)

    window["-IMAGE-"].update(data=array_to_data(img))

    window.refresh()
    window["-VIEWER-"].contents_changed()


def write_stats(window, stats):
    if stats is None:
        window["-MACRO_F1-"].update(f"Macro F1: -")
        window["-MACRO_F1_DENOISED-"].update(f"Macro F1 (denoised): -")
        window["-PPA-"].update(f"Per pixel accuracy: -")
        window["-PPA_DENOISED-"].update(f"Per pixel accuracy (denoised): -")
        window["-MACRO_F1_BY_CLASS-"].update("Macro F1 by class (F1 / Denoised F1):\n-")
        window["-MACRO_F1_BY_AREA-"].update("Macro F1 by area (F1 / Denoised F1):\n-")
        return

    window["-MACRO_F1-"].update(f"Macro F1: {stats[MACRO_F1]:.2%}")
    window["-MACRO_F1_DENOISED-"].update(
        f"Macro F1 (denoised): {stats[MACRO_F1_DENOISED]:.2%}"
    )

    window["-PPA-"].update(f"Per pixel accuracy: {stats[PPA]:.2%}")
    window["-PPA_DENOISED-"].update(
        f"Per pixel accuracy (denoised): {stats[PPA_DENOISED]:.2%}"
    )

    f1_by_class = ["Macro F1 by class (F1 / Denoised F1):"]
    for class_name, f1 in stats[F1_BY_CLASS].items():
        frmt_class_name = class_name + ":" + " " * max(0, 6 - len(class_name))
        line = f"{frmt_class_name}\t{f1:.1%} / {stats[F1_BY_CLASS_DENOISED][class_name]:.1%}"
        f1_by_class.append(line)

    window["-MACRO_F1_BY_CLASS-"].update("\n".join(f1_by_class))

    f1_by_area = ["Macro F1 by area (F1 / Denoised F1):"]
    for area_name, f1 in stats[F1_BY_AREA].items():
        frmt_area_name = area_name + ":" + " " * max(0, 10 - len(area_name))
        line = (
            f"{frmt_area_name}\t{f1:.1%} / {stats[F1_BY_AREA_DENOISED][area_name]:.1%}"
        )
        f1_by_area.append(line)

    window["-MACRO_F1_BY_AREA-"].update("\n".join(f1_by_area))

    window.refresh()
    window["-VIEWER-"].contents_changed()


def filter_results(
    results,
    cur_experiment: str,
    cur_origin: str,
    patch_type: str,
    cur_areas: str,
    caches_stats,
):
    if patch_type == MAIN_ONLY:
        results = results.filter(lambda result: not result.is_corner_patch)
    elif patch_type == CORNER_ONLY:
        results = results.filter(lambda result: result.is_corner_patch)

    if cur_areas == BEST or cur_areas == WORST:
        if cur_experiment not in caches_stats[cur_origin]:
            caches_stats[cur_origin][cur_experiment] = {
                patch_type: None,
            }
        elif patch_type not in caches_stats[cur_origin][cur_experiment]:
            caches_stats[cur_origin][cur_experiment][patch_type] = None

        stats = caches_stats[cur_origin][cur_experiment][patch_type]

        if stats is None:
            stats = calc_stats(results)
            caches_stats[cur_origin][cur_experiment][patch_type] = stats

        if cur_areas == BEST:
            results = results.filter(
                lambda result: result.brain_area in stats[F1_BEST_CUTOFF]
            )
        elif cur_areas == WORST:
            results = results.filter(
                lambda result: result.brain_area in stats[F1_WORST_CUTOFF]
            )

    return results


def draw_plot(results, seed, to_display, n_patches: int, target_size=IMAGE_SIZE):
    np.random.seed(seed)

    n_plots_per_patch = np.sum(list(to_display.values()))

    n_patches_per_row = max(8 // n_plots_per_patch, 1)
    n_columns = n_patches_per_row * n_plots_per_patch
    n_rows = np.ceil(n_patches / n_patches_per_row).astype(int)

    sample_results = results.sample(n_patches)

    fig, ax = plt.subplots(n_rows, n_columns, figsize=(n_columns * 2, n_rows * 5))

    if n_rows == 1:
        ax = ax[np.newaxis, :]

    for i, rand_result in enumerate(sample_results):
        offset = 0
        cur_ax_row = i // n_patches_per_row
        cur_ax_col = (i % n_patches_per_row) * n_plots_per_patch

        for key in to_display:
            if to_display[key]:
                if key == INPUT_IMAGE:
                    ax[cur_ax_row, cur_ax_col + offset].imshow(
                        rand_result.input_image, cmap="gray"
                    )
                    ax[cur_ax_row, cur_ax_col + offset].set_title(
                        f"{rand_result.brain_area}"
                    )
                    ax[cur_ax_row, cur_ax_col + offset].axis("off")
                elif key == GT_SEGMENTATION:
                    img = cort.display.colour_patch(
                        rand_result.input_image, rand_result.gt_segmentation
                    )
                    ax[cur_ax_row, cur_ax_col + offset].imshow(img)
                    ax[cur_ax_row, cur_ax_col + offset].set_title(
                        f"{rand_result.brain_area}\nGT Segmentation"
                    )
                    ax[cur_ax_row, cur_ax_col + offset].axis("off")
                elif key == PRED_SEGMENTATION_SHOW_ERRORS:
                    evaluate.display_patch_diff_alt(
                        rand_result.input_image,
                        rand_result.gt_segmentation,
                        rand_result.pred_segmentation,
                        ax[cur_ax_row, cur_ax_col + offset],
                    )
                    ax[cur_ax_row, cur_ax_col + offset].set_title(
                        f"{rand_result.brain_area}\nPrediction - {rand_result.prediction.accuracy:.0%}"
                    )
                    ax[cur_ax_row, cur_ax_col + offset].axis("off")
                elif key == PREV_DENOISED_SEGMENTATION_SHOW_ERRORS:
                    if rand_result.pred_denoised_segmentation is not None:
                        evaluate.display_patch_diff_alt(
                            rand_result.input_image,
                            rand_result.gt_segmentation,
                            rand_result.pred_denoised_segmentation,
                            ax[cur_ax_row, cur_ax_col + offset],
                        )
                    ax[cur_ax_row, cur_ax_col + offset].set_title(
                        f"{rand_result.brain_area}\nDenoised"
                    )
                    ax[cur_ax_row, cur_ax_col + offset].axis("off")
                elif key == PRED_DEPTH_MAP:
                    if rand_result.pred_depth_map is not None:
                        img = cort.display.draw_max_depth(
                            rand_result.pred_depth_map, rand_result.input_image
                        )
                        ax[cur_ax_row, cur_ax_col + offset].imshow(img)
                    ax[cur_ax_row, cur_ax_col + offset].set_title(
                        f"{rand_result.brain_area}\nDepth Map"
                    )
                    ax[cur_ax_row, cur_ax_col + offset].axis("off")
                elif key == AUTOENCODED_IMG:
                    img = rand_result.autoencoded_img
                    if img is not None:
                        ax[cur_ax_row, cur_ax_col + offset].imshow(img, cmap="gray")
                    ax[cur_ax_row, cur_ax_col + offset].set_title(
                        f"{rand_result.brain_area}\nAutoencoded Image"
                    )
                    ax[cur_ax_row, cur_ax_col + offset].axis("off")
                elif key == PRED_SEGMENTATION:
                    img = cort.display.colour_patch(
                        rand_result.input_image, rand_result.pred_segmentation
                    )
                    ax[cur_ax_row, cur_ax_col + offset].imshow(img)
                    ax[cur_ax_row, cur_ax_col + offset].set_title(
                        f"{rand_result.brain_area}\nPrediction - {rand_result.prediction.accuracy:.0%}"
                    )
                    ax[cur_ax_row, cur_ax_col + offset].axis("off")
                elif key == PRED_DENOISED_SEGMENTATION:
                    img = cort.display.colour_patch(
                        rand_result.input_image, rand_result.pred_denoised_segmentation
                    )
                    ax[cur_ax_row, cur_ax_col + offset].imshow(img)
                    ax[cur_ax_row, cur_ax_col + offset].set_title(
                        f"{rand_result.brain_area}\nDenoised Prediction"
                    )
                    ax[cur_ax_row, cur_ax_col + offset].axis("off")
                offset += 1

    for i in range(n_patches_per_row * n_rows - n_patches):
        for j in range(n_plots_per_patch):
            ax[n_rows - 1, -(i + 1) * n_plots_per_patch + j].axis("off")

    fig.tight_layout()
    fig.canvas.draw()

    data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))

    plt.close(fig)

    return data


def update_train_plot(train_log, window):
    fig, _ = train_log.plot(smoothing=100, show=False)
    fig.canvas.draw()

    data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))

    window["-TRAIN_PLOT-"].update(data=array_to_data(data))

    plt.close(fig)


def calc_stats(results):
    macro_f1 = experiments.calc_macro_f1(results)
    macro_f1_denoised = experiments.calc_macro_f1(results, denoised=True)

    f1_by_class = experiments.calc_f1_by_class(results)
    f1_by_class_denoised = experiments.calc_f1_by_class(results, denoised=True)

    macro_f1_by_area = experiments.calc_f1_by_brain_area(results)
    macro_f1_by_area_denoised = experiments.calc_f1_by_brain_area(
        results, denoised=True
    )

    f1_best_cutoff = np.quantile(np.array(list(macro_f1_by_area.values())), 0.8)
    f1_worst_cutoff = np.quantile(np.array(list(macro_f1_by_area.values())), 0.2)

    f1_best_cutoff = [
        area for area, f1 in macro_f1_by_area.items() if f1 >= f1_best_cutoff
    ]
    f1_worst_cutoff = [
        area for area, f1 in macro_f1_by_area.items() if f1 <= f1_worst_cutoff
    ]

    return {
        MACRO_F1: macro_f1,
        MACRO_F1_DENOISED: macro_f1_denoised,
        F1_BY_CLASS: f1_by_class,
        F1_BY_CLASS_DENOISED: f1_by_class_denoised,
        F1_BY_AREA: macro_f1_by_area,
        F1_BY_AREA_DENOISED: macro_f1_by_area_denoised,
        F1_BEST_CUTOFF: f1_best_cutoff,
        F1_WORST_CUTOFF: f1_worst_cutoff,
    }


def array_to_data(array):
    im = Image.fromarray(array)
    with BytesIO() as output:
        im.save(output, format="PNG")
        data = output.getvalue()
    return data


if __name__ == "__main__":
    main()
