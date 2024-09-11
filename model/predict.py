import numpy as np
from implementations import sigmoid


def predict(
    data: np.ndarray, w: np.ndarray, threshold: float, negative_label: int = 0
) -> np.ndarray:
    """
    Predict the labels of the data using the weights w.
    Note: remember to use data with offset column if the
    weights w have been trained with offset column.

    :param data: data for which to predict the labels
    :param w: optimal weights
    :param threshold: threshold for the label decision
    :param negative_label: label to assign to the negative class

    :return: predicted labels
    """

    return np.where(sigmoid(data @ w) >= threshold, 1, negative_label)


def predict_no_labels(data: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Predict the output (NOT LABELS) of the data using the weights w.
    Note: remember to use data with offset column if the
    weights w have been trained with offset column.

    :param data: data for which to predict the labels
    :param w: optimal weights

    :return: predicted labels
    """

    return sigmoid(data @ w)
