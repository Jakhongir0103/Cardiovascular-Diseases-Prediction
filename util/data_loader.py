from typing import List, Union
import numpy as np
import csv


def load_dataset(
    path_x_train: str, path_y_train: str, path_x_test: str, sub_sample: bool = False
) -> (np.ndarray, np.ndarray, np.ndarray, List[str]):
    """
    Loads the dataset
    :param path_x_train: path of x_train.csv
    :param path_y_train: path of y_train.csv
    :param path_x_test: path of x_test.csv
    :param sub_sample: if True, keep only 50 rows of the dataset
    :return: (x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, feature_names: List)
    """

    features_names = np.genfromtxt(path_x_train, delimiter=",", dtype=str, max_rows=1)

    x_train = np.genfromtxt(path_x_train, delimiter=",", skip_header=1)

    y_train = np.genfromtxt(path_y_train, delimiter=",", skip_header=1)

    x_test = np.genfromtxt(path_x_test, delimiter=",", skip_header=1)

    # sub-sample
    if sub_sample:
        x_train = x_train[::50]
        y_train = y_train[::50]
        x_test = x_test[::50]

    return x_train, y_train, x_test, features_names


def create_csv_submission(ids: Union[List, np.ndarray], y_pred: np.ndarray, path: str):
    """
    This function creates a csv file named 'name' in the format required for a submission in Kaggle or AIcrowd.
    The file will contain two columns the first with 'ids' and the second with 'y_pred'.
    y_pred must be a list or np.ndarray of 1 and -1 otherwise the function will raise a ValueError.

    Args:
        ids (list,np.array): indices
        y_pred (list,np.array): predictions on data correspondent to indices
        name (str): name of the file to be created
    """
    # Check that y_pred only contains -1 and 1
    if not all(i in [-1, 1] for i in y_pred):
        raise ValueError("y_pred can only contain values -1, 1")

    with open(path, "w", newline="") as csvfile:
        fieldnames = ["Id", "Prediction"]
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({"Id": int(r1), "Prediction": int(r2)})


def split_train_validation(
    x: np.ndarray, y: np.ndarray, valid_proportion: float
) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    valid_index = np.random.choice(
        np.arange(x.shape[0]), size=int(x.shape[0] * valid_proportion), replace=False
    )
    x_train = x[~np.isin(np.arange(x.shape[0]), valid_index), :]
    x_valid = x[valid_index, :]

    if y.ndim == 2:
        y_train = y[~np.isin(np.arange(x.shape[0]), valid_index), :]
        y_valid = y[valid_index, :]
    else:
        assert y.ndim == 1
        y_train = y[~np.isin(np.arange(x.shape[0]), valid_index)]
        y_valid = y[valid_index]

    return x_train, x_valid, y_train, y_valid


def custom_random_oversampling(x, y):
    # Separate the majority and minority classes
    majority_class = x[y == 0, :]
    minority_class = x[y == 1, :]

    # Find the number of instances in each class
    num_majority = len(majority_class)
    num_minority = len(minority_class)

    # Calculate the oversampling ratio
    oversample_ratio = num_majority // num_minority

    # Randomly duplicate instances from the minority class
    oversampled_minority_index = np.random.choice(
        minority_class.shape[0],
        size=num_minority * oversample_ratio - num_minority,
        replace=True,
    )
    oversampled_minority = minority_class[oversampled_minority_index, :]

    # Combine the oversampled minority class with the majority class
    x_resampled = np.vstack((majority_class, minority_class, oversampled_minority))
    y_resampled = np.hstack(
        (np.zeros(num_majority), np.ones(num_minority + oversampled_minority.shape[0]))
    )

    return x_resampled, y_resampled


def change_negative_class(y: np.ndarray, current: int, new: int):
    """
    Replace a specified class label in an array with a new class label.

    Args:
    - y (numpy.ndarray): An array containing the original class labels.
    - current: The class label you want to replace in the array.
    - new: The new class label to replace the 'current' label.

    Returns:
    - numpy.ndarray: A new array with the same shape as 'y,' where occurrences of the 'current' label have been replaced with the 'new' label.

    """
    return np.where(y == current, new, y)
