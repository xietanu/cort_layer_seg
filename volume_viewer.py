import ctypes
import json
import os
import time
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
import volume

DATA_BASE_PATH = "data/volumes"

IMAGE_SIZE = (1200, 1200)

SELECTED = "-SELECTED-"
IMAGE = "-IMAGE-"

X_CUTOUT = "-X_CUTOUT-"
Y_CUTOUT = "-Y_CUTOUT-"
Z_CUTOUT = "-Z_CUTOUT-"

X_ROT = "-X_ROT-"
Y_ROT = "-Y_ROT-"
Z_ROT = "-Z_ROT-"


UPDATE_INTERVAL = 0.1


def main():
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
    cur_volume_fn = None
    cur_volume = None
    cur_cutout = None
    cur_shape = None
    cur_x_cutoff = 0.5
    cur_y_cutoff = 0.5
    cur_z_cutoff = 0.5
    cur_x_rot = np.pi
    cur_y_rot = np.pi
    cur_z_rot = np.pi

    update_countdown = -1

    sg.theme("DarkBlue14")

    volume_fns = os.listdir("data/volumes")

    sidebar = [
        [
            sg.Text("Experiment:"),
            sg.DropDown(volume_fns, key=SELECTED, enable_events=True),
        ],
        [sg.HSeparator()],
        [sg.Text("X rotation:")],
        [
            sg.Slider(
                range=(0, 100),
                default_value=50,
                orientation="horizontal",
                key=X_ROT,
                enable_events=True,
            )
        ],
        [sg.Text("Y rotation:")],
        [
            sg.Slider(
                range=(0, 100),
                default_value=50,
                orientation="horizontal",
                key=Y_ROT,
                enable_events=True,
            )
        ],
        [sg.Text("Z rotation:")],
        [
            sg.Slider(
                range=(0, 100),
                default_value=50,
                orientation="horizontal",
                key=Z_ROT,
                enable_events=True,
            )
        ],
        [sg.HSeparator()],
        [sg.Text("X cutout:")],
        [
            sg.Slider(
                range=(0, 100),
                default_value=50,
                orientation="horizontal",
                key=X_CUTOUT,
                enable_events=True,
            )
        ],
        [sg.Text("Y cutout:")],
        [
            sg.Slider(
                range=(0, 100),
                default_value=50,
                orientation="horizontal",
                key=Y_CUTOUT,
                enable_events=True,
            )
        ],
        [sg.Text("Z cutout:")],
        [
            sg.Slider(
                range=(0, 100),
                default_value=50,
                orientation="horizontal",
                key=Z_CUTOUT,
                enable_events=True,
            )
        ],
    ]

    main_viewer = [
        [
            sg.Image(key="-IMAGE-", size=IMAGE_SIZE),
        ],
    ]

    layout = [
        [
            sg.Column(sidebar),
            sg.VSeparator(),
            sg.Column(
                main_viewer,
                scrollable=True,
                vertical_scroll_only=True,
                size=IMAGE_SIZE,
                key="-VIEWER-",
            ),
        ],
    ]

    # Create the window

    window = sg.Window("Volume Viewer", layout, resizable=True, finalize=True)

    window[IMAGE].update(
        data=array_to_data(np.zeros((IMAGE_SIZE[1], IMAGE_SIZE[0], 4), dtype=np.uint8))
    )

    last_updated = time.time()

    # Create an event loop
    while True:
        dt = time.time() - last_updated
        last_updated = time.time()

        event, values = window.read(timeout=int(UPDATE_INTERVAL * 100))
        # End program if user closes window or
        # presses the OK button
        if event == "OK" or event == sg.WIN_CLOSED:
            break

        if event == SELECTED:
            new_volume_fn = values[SELECTED]
            if cur_volume_fn != new_volume_fn:
                cur_volume_fn = new_volume_fn

                cur_volume = np.load(f"{DATA_BASE_PATH}/{cur_volume_fn}")
                cur_shape = cur_volume.shape

                pad_amount = int(cur_shape[0] * 0.1)

                col_vol_sml = np.zeros_like(cur_volume)
                col_vol_sml[
                    pad_amount:-pad_amount,
                    pad_amount:-pad_amount,
                    pad_amount:-pad_amount,
                ] = cur_volume[
                    pad_amount:-pad_amount,
                    pad_amount:-pad_amount,
                    pad_amount:-pad_amount,
                ]
                col_vol_sml = np.pad(
                    col_vol_sml,
                    (
                        (pad_amount * 2, pad_amount * 2),
                        (pad_amount * 2, pad_amount * 2),
                        (pad_amount * 2, pad_amount * 2),
                        (0, 0),
                    ),
                )
                cur_volume = col_vol_sml

                cur_volume = volume.prerender_tensor(cur_volume)
                update_countdown = 0.01

        if event == X_CUTOUT:
            cur_x_cutoff = (100 - values[X_CUTOUT]) / 100
            update_countdown = UPDATE_INTERVAL

        if event == Y_CUTOUT:
            cur_y_cutoff = (100 - values[Y_CUTOUT]) / 100
            update_countdown = UPDATE_INTERVAL

        if event == Z_CUTOUT:
            cur_z_cutoff = (100 - values[Z_CUTOUT]) / 100
            update_countdown = UPDATE_INTERVAL

        if event == X_ROT:
            cur_x_rot = 2 * np.pi * (100 - values[X_ROT]) / 100
            update_countdown = UPDATE_INTERVAL

        if event == Y_ROT:
            cur_y_rot = 2 * np.pi * (100 - values[Y_ROT]) / 100
            update_countdown = UPDATE_INTERVAL

        if event == Z_ROT:
            cur_z_rot = 2 * np.pi * (100 - values[Z_ROT]) / 100
            update_countdown = UPDATE_INTERVAL

        if update_countdown > 0:
            update_countdown -= dt
            if update_countdown <= 0:
                update(
                    cur_volume,
                    cur_cutout,
                    cur_shape,
                    cur_x_rot,
                    cur_y_rot,
                    cur_z_rot,
                    cur_x_cutoff,
                    cur_y_cutoff,
                    cur_z_cutoff,
                    window,
                )

    window.close()


def update(
    cur_volume,
    cur_cutout,
    cur_shape,
    x_rot,
    y_rot,
    z_rot,
    x_cutoff,
    y_cutoff,
    z_cutoff,
    window,
):
    if cur_volume is None:
        return

    img = volume.render_tensor(
        cur_volume,
        angles=(x_rot - np.pi / 4, y_rot - 3 * np.pi / 4, z_rot),
        cutout_perc=(x_cutoff, y_cutoff, z_cutoff),
    )

    x_ratio = img.shape[1] / IMAGE_SIZE[0]
    y_ratio = img.shape[0] / IMAGE_SIZE[1]

    ratio = max(x_ratio, y_ratio)

    img = cv2.resize(img, (int(img.shape[1] / ratio), int(img.shape[0] / ratio)))

    x_pad = (IMAGE_SIZE[0] - img.shape[1]) // 2
    y_pad = (IMAGE_SIZE[1] - img.shape[0]) // 2

    img = np.pad(
        img,
        ((y_pad, y_pad), (x_pad, x_pad), (0, 0)),
        mode="constant",
        constant_values=0,
    )

    window["-IMAGE-"].update(data=array_to_data(img))

    window.refresh()
    window["-VIEWER-"].contents_changed()


def array_to_data(array):
    im = Image.fromarray(array)
    with BytesIO() as output:
        im.save(output, format="PNG")
        data = output.getvalue()
    return data


if __name__ == "__main__":
    main()
