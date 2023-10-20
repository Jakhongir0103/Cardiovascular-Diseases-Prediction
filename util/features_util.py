
from typing import Union, Dict, List, Callable
from util.features_info import Feature, FeatureType, FEATURES_DICT

import numpy as np


def drop_features(data: np.ndarray, feature_to_drop: Union[str, list[str]], features: list[str], feature_index: dict):
    """
    Drop a feature for all the samples.
    :param data: np.array of shape (N, D)
    :param feature_to_drop: list(str) | str Name(s) of the feature(s) to drop
    :param features: list(str) containing the names of the features currently in the data
    :param feature_index: dict(str : int) linking the name of the features to their index

    :return: data, features, feature_index updated after dropping the feature(s) specified in feature_to_drop
    """

    # Get the index of the feature to drop
    ids_feature_to_drop = [feature_index[feature] for feature in feature_to_drop]

    # Drop the column corresponding to the feature to drop and update the features list
    data = np.delete(data, ids_feature_to_drop, axis=1)
    features = [f for f in features if f not in feature_to_drop]

    # Update the feature_index dictionary
    assert len(features) == data.shape[1]
    feature_index = {feature: index for index, feature in enumerate(features)}

    print(f"Removed {len(feature_to_drop)} features: {feature_to_drop}")

    return data, features, feature_index


def keep_features(data: np.ndarray, features_to_keep: Union[str, list[str]], features: list[str], feature_index: dict):
    """
    Keep only the feature(s) specified in features_to_keep.
    :param data: np.array of shape (N, D)
    :param features_to_keep: name(s) of the feature(s) to keep
    :param features: list(str) of the names of the features
    :param feature_index: dict(str : int) linking the name of the features to their index
    """

    # Get the index of the feature to keep
    ids_features_to_keep = [feature_index[feature] for feature in features_to_keep]

    # Keep only the columns corresponding to the feature to keep and update the features list
    data = data[:, ids_features_to_keep]
    features = features_to_keep

    # Update the feature_index dictionary
    assert len(features) == data.shape[1]
    feature_index = {feature: index for index, feature in enumerate(features)}

    print(f"Kept {len(features_to_keep)} features: {features_to_keep}")

    return data, features, feature_index


def drop_feature_threshold(data: np.ndarray, features: list[str], feature_index: dict, threshold=0.9):
    """
    Drop feature for all the samples, if the values for that features are NaN
    for a percentage higher than threshold.
    :param data: np.array of shape (N, D)
    :param features: list(str) of the names of the features
    :param feature_index: dict(str : int) linking the name of the features to their index
    :param threshold: percentage of NaN values above which we drop the feature
    """

    N = data.shape[0]

    # Compute percentage of NaN values for each feature
    n_NaN = np.sum(np.isnan(data), axis=0)
    p_NaN = n_NaN / N

    # Get the indices of the features for which the percentage of NaN values is higher than threshold
    ids_features_to_drop = np.where(p_NaN > threshold)[0]
    features_to_drop = []
    for f in feature_index.keys():
        if feature_index[f] in ids_features_to_drop:
            features_to_drop.append(f)
    assert len(features_to_drop) == len(ids_features_to_drop)

    return drop_features(data, features_to_drop, features, feature_index)


def set_nans_to_value(x: np.ndarray, value: int, where: Union[str, List[str]],
                      feature_index: Dict[str, int]) -> np.ndarray:
    """
    Maps all the nan values to a specified value.
    Note: not inplace, dataset is copied.

    :param x: dataset
    :param value: replacement value for nan
    :param where: feature(s) where to apply the mapping operation
    :param feature_index: dictionary that maps feature names to the (column) index in the dataset x
    :return: (new) dataset after the preprocessing
    """
    vectorized_nans_set = np.vectorize(lambda feature, v: value if np.isnan(v) else v)
    return _apply_preprocessing(x, where, feature_index, vectorized_operation=vectorized_nans_set)


def align_nans(x: np.ndarray, where: Union[str, List[str]], feature_index: Dict[str, int]) -> np.ndarray:
    """
    Maps all the nan aliases to np.nan.
    Note: not inplace, dataset is copied!

    :param x: dataset
    :param where: feature(s) where to apply the mapping operation
    :param feature_index: dictionary that maps feature names to the (column) index in the dataset x
    :return: (new) dataset after the preprocessing
    """
    vectorized_nan_replacing = np.vectorize(lambda feature, v: v if not feature.isnan(v) else np.nan)
    return _apply_preprocessing(x, where, feature_index, vectorized_nan_replacing)


def map_values(x: np.ndarray, where: Union[str, List[str]], feature_index: Dict[str, int]):
    """
    Map values of feature(s) according to the pre-determined mapping.
    Note: not inplace, dataset is copied!

    :param x: dataset
    :param where: feature(s) where to apply the mapping operation
    :param feature_index: dictionary that maps feature names to the (column) index in the dataset x
    :return: (new) dataset after the preprocessing
    """

    vectorized_remapping = np.vectorize(lambda feature, v: feature.map_values[v] if v in feature.map_values else v)
    return _apply_preprocessing(x, where, feature_index, vectorized_remapping)


def _apply_preprocessing(x: np.ndarray, where: Union[str, List[str]], feature_index: Dict[str, int],
                         vectorized_operation: Callable) -> np.ndarray:
    """
    Skeleton for vectorized-custom-preprocessing of the dataset.
    Note: not inplace, dataset is copied!

    :param x: dataset
    :param where: feature(s) where to apply the preprocessing operation
    :param feature_index: dictionary that maps feature names to the (column) index in the dataset x
    :param vectorized_operation: a vectorized operation to apply for the specified features
    :return: (new) dataset after the preprocessing
    """

    if type(where) == str:
        where = [where]

    x_processed = x.copy()

    for feature_name in where:
        assert feature_name in feature_index
        assert feature_name in FEATURES_DICT
        idx = feature_index[feature_name]
        feature = FEATURES_DICT[feature_name]

        x_processed[:, idx] = vectorized_operation(feature, x_processed[:, idx])

    return x_processed
