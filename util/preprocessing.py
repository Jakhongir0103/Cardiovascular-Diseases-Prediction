from typing import Union, Dict, List, Callable
from util.features_info import Feature, FeatureType, FEATURES_DICT, REPLACEMENT_DICT

import numpy as np


def remove_nans(x: np.ndarray,
                where: Union[str, List[str]],
                feature_index: Dict[str, int]) -> np.ndarray:
    """
    Call set_nans_to_value for all the specified features using the correct value,
    computed here if needed (for example for mean or median).
    Note: not inplace, dataset is copied!

    :param x: dataset
    :param where: feature(s) where to apply the mapping operation
    :param feature_index: dictionary that maps feature names to the (column) index in the dataset x
    :return: (new) dataset after the preprocessing
    """
    # TODO: this method must be called after the align_nans method
    # NOTE: with this implementation the dataset may cause multiple copies
    # because of multiple calls to set_nans_to_value. Solve this issue.
    if type(where) == str:
        where = [where]

    for f in feature_index.keys():
        value = REPLACEMENT_DICT[f]
        if type(value) == str:
            if value == 'mean':
                value = np.nanmean(x[:, feature_index[f]])
            elif value == 'median':
                value = np.nanmedian(x[:, feature_index[f]])
        elif type(value) == int:
            pass
        else:
            raise ValueError(f"Unknown value {value} for feature {f}")

        x = set_nans_to_value(x, value, f, feature_index)

    return x


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

    vectorized_remapping = np.vectorize(lambda feature, v:
                                        feature.map_values[v] if not np.isnan(v) and v in feature.map_values else v,
                                        otypes=[np.float64])
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
