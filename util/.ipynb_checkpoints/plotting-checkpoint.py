from typing import Union, List, Dict

import matplotlib.pyplot as plt
import numpy as np


def multiple_hists(
    data: np.ndarray, columns: Union[str, List[str]], feature_index: Dict[str, int]
):
    if type(columns) == str:
        columns = [columns]

    num_plots_side = int(np.ceil(np.sqrt(len(columns))))
    fig, axes = plt.subplots(num_plots_side, num_plots_side, figsize=(14, 14))
    for n, column in enumerate(columns):
        ax = axes[n // num_plots_side][n % num_plots_side]
        ax.hist(data[:, feature_index[column]])
        ax.set_xlabel(column)


def correlation_matrix_dist(
    data: np.ndarray, columns: Union[str, List[str]], feature_index: Dict[str, int]
):
    """
    Prints a square plot that shows a correlation heatmap above the diagonal,
    a scatter plot with linear regression below the diagonal, and distribution
    of the data on the diagonal. Note: the color bar is still missing!

    :param data: data to be used
    :param columns: list of the features of the data
    :param feature_index: dictionary to retrieve the index corresponding to each feature
    """
    n = len(columns)
    if n > 30:
        print("Warning: number of columns really high!")

    if type(columns) == str:
        columns = [columns]

    fig, axs = plt.subplots(
        n,
        n,
        figsize=(16, 16),
    )

    nan_mask = np.isnan(data).any(axis=1)
    clean_data = data[~nan_mask]

    for i in range(n):
        for j in range(n):
            x = clean_data[feature_index[columns[i]]]
            y = clean_data[feature_index[columns[j]]]
            if i == j:
                axs[i, j].hist(data[:, feature_index[columns[i]]])
            elif i > j:
                axs[i, j].plot(x, y, "o")
                m, b = np.polyfit(x, y, 1)
                axs[i, j].plot(x, m * x + b)
            else:
                axs[i, j].imshow(
                    [[np.corrcoef(x, y, rowvar=False)[0][1]]],
                    vmin=0,
                    vmax=1,
                    cmap="RdBu",
                )
                axs[i, j].grid(None)
                axs[i, j].xaxis.set_tick_params(labelbottom=False)
                axs[i, j].yaxis.set_tick_params(labelleft=False)
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])

            if i == 0:
                axs[i, j].set_title(columns[j], fontsize=14)
            if j == 0:
                axs[i, j].set_ylabel(columns[i], fontsize=14)


def loss_visualization(
    train_loss: np.ndarray,
    validation_loss: np.ndarray,
):
    """
    Plots the loss for the train and validation sets.
    :param train_loss: loss for the train set
    :param validation_loss: loss for the validation set
    """
    plt.figure(figsize=(14, 7))

    plt.plot(train_loss, marker=".", color="b", label="train loss")
    plt.plot(validation_loss, marker=".", color="r", label="validation loss")

    plt.xlabel("iter")
    plt.ylabel("loss")
    plt.title("Loss")
    plt.legend(loc=2)
    plt.grid(True)


def confusion_matrix(true_labels: np.ndarray, predicted_labels: np.ndarray):
    """
    Plots the confusion matrix.
    :param train_loss: loss for the train set
    :param validation_loss: loss for the validation set
    """
    # Calculate the confusion matrix
    classes = [1, 0]
    cm = np.zeros((len(classes), len(classes)), dtype=float)

    for i_idx, i in enumerate(classes):
        for j_idx, j in enumerate(classes):
            cm[i_idx, j_idx] = (
                100
                * np.sum((true_labels == i) & (predicted_labels == j))
                / len(true_labels)
            )

    # Create the confusion matrix heatmap
    plt.figure(figsize=(8, 6))
    cax = plt.matshow(cm, cmap="Blues")

    for i in range(len(classes)):
        for j in range(len(classes)):
            plt.text(
                j,
                i,
                f"{cm[i, j]:.2f}%",
                va="center",
                ha="center",
                color="black",
                fontsize=12,
            )

    plt.xticks(range(len(classes)), classes)
    plt.yticks(range(len(classes)), classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()
