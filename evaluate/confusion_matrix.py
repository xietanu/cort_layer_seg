import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

import cort


def confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, padding_value: int | None = None
) -> np.ndarray:
    """
    Compute confusion matrix to evaluate the accuracy of a classification.
    """
    # if image, flatten
    if len(y_true.shape) == 3:
        y_true = np.reshape(y_true, (y_true.shape[0], -1))
        y_pred = np.reshape(y_pred, (y_pred.shape[0], -1))

    labels = np.unique(y_true)

    if padding_value is not None:
        labels = labels[labels != padding_value]

    n_labels = len(labels)

    conf_mat = np.zeros((n_labels, n_labels))

    for i, true_label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            conf_mat[i, j] = np.sum((y_true == true_label) & (y_pred == pred_label))

    conf_mat = conf_mat / np.sum(conf_mat, axis=1)[:, None]

    return conf_mat


def show_confusion_matrix(
    true_masks: np.ndarray,
    pred_masks: np.ndarray,
    title: str,
    padding_value=cort.constants.PADDING_MASK_VALUE,
):
    conf_mat = confusion_matrix(true_masks, pred_masks, padding_value=padding_value)

    labels = ["BG", "I", "II", "III", "IV", "V", "VI", "WM"]

    conf_mat = pd.DataFrame(conf_mat, index=labels, columns=labels)

    sns.set_theme(font_scale=1.4)  # for label size
    ax = sns.heatmap(
        conf_mat, annot=True, annot_kws={"size": 12}, fmt=".0%"
    )  # font size
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    plt.show()
