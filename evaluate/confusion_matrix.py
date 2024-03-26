import numpy as np


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
