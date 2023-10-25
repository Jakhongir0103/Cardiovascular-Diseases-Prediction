from typing import Union, Dict, List, Callable, Tuple
from util.features_info import Feature, FeatureType, FEATURES_DICT, REPLACEMENT_DICT

import numpy as np


def preprocessing_pipeline(x: np.ndarray,
                           where: Union[str, List[str]],
                           feature_index: Dict[str, int],
                           nan_replacement: List[Tuple[List[str], Union[float, str]]] = None) -> np.ndarray:
    prepro_df = align_nans(map_values(x, where, feature_index), where, feature_index)
    if nan_replacement is not None:
        for (features_list, val) in nan_replacement:
            prepro_df = set_nans_to_value(prepro_df, value=val, where=features_list, feature_index=feature_index)

    return prepro_df



def set_nans_to_value(x: np.ndarray, value: Union[float, str], where: Union[str, List[str]],
                      feature_index: Dict[str, int]) -> np.ndarray:
    """
    Maps all the nan values to a specified value.
    Note: not inplace, dataset is copied.

    :param x: dataset
    :param value: replacement value for nan, or alternatively "mean", "median"
    :param where: feature(s) where to apply the mapping operation
    :param feature_index: dictionary that maps feature names to the (column) index in the dataset x
    :return: (new) dataset after the preprocessing
    """

    def nan_replacer(feature: Feature, v: np.ndarray):
        """
        Replaces nan values in v (value is given)
        :param feature: unused
        :param v: 1D array, column where to replace nan values
        :return: column with replaced nan values
        """
        if type(value) == str:
            if value == "mean":
                replacement = np.nanmean(v)
            elif value == "median":
                replacement = np.nanmedian(v)
            else:
                raise AssertionError("Invalid value for replacement")
        else:
            replacement = value

        return np.nan_to_num(v, nan=replacement)

    return _apply_preprocessing(x, where, feature_index, vectorized_operation=nan_replacer)


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