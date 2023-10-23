from typing import Union, Dict, List
import numpy as np


def drop_features(
    data: np.ndarray,
    feature_to_drop: Union[str, list[str]],
    features: list[str],
    feature_index: dict,
):
    """
    Drop a feature for all the samples.
    :param data: np.array of shape (N, D)
    :param feature_to_drop: list(str) | str Name(s) of the feature(s) to drop
    :param features: list(str) containing the names of the features currently in the data
    :param feature_index: dict(str : int) linking the name of the features to their index

    :return: data, features, feature_index updated after dropping the feature(s) specified in feature_to_drop
    """

    if type(feature_to_drop) == str:
        feature_to_drop = [feature_to_drop]

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


def keep_features(
    data: np.ndarray,
    features_to_keep: Union[str, list[str]],
    features: list[str],
    feature_index: dict,
):
    """
    Keep only the feature(s) specified in features_to_keep.
    :param data: np.array of shape (N, D)
    :param features_to_keep: name(s) of the feature(s) to keep
    :param features: list(str) of the names of the features
    :param feature_index: dict(str : int) linking the name of the features to their index
    """

    if type(features_to_keep) == str:
        features_to_keep = [features_to_keep]

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


def drop_feature_threshold(
    data: np.ndarray, features: list[str], feature_index: dict, threshold=0.9
):
    """
    Drop feature for all the samples, if the values for that features are NaN
    for a percentage higher than threshold.
    :param data: np.array of shape (N, D)
    :param features: list(str) of the names of the features
    :param feature_index: dict(str : int) linking the name of the features to their index
    :param threshold: percentage of NaN values above which we drop the feature
    """

    n = data.shape[0]

    # Compute percentage of NaN values for each feature
    n_nan = np.sum(np.isnan(data), axis=0)
    p_nan = n_nan / n

    # Get the indices of the features for which the percentage of NaN values is higher than threshold
    ids_features_to_drop = np.where(p_nan > threshold)[0]
    features_to_drop = []
    for f in feature_index.keys():
        if feature_index[f] in ids_features_to_drop:
            features_to_drop.append(f)
    assert len(features_to_drop) == len(ids_features_to_drop)

    return drop_features(data, features_to_drop, features, feature_index)


def keep_uncorrelated_features(
    data: np.ndarray,
    features: Union[list[str], str],
    feature_index: Dict[str, int],
    threshold: float = 0.9,
):
    """
    Keep only the features that are not highly correlated, i.e. the features
    that have a correlation coefficient lower than the threshold.
    **Note**: to compute the correlation coefficient, we remove the samples
    that contain NaN values, so to avoid losing too much data, we **first
    clean the data with other preprocessing methods**.

    :param data: np.array of shape (N, D)
    :param features: list(str) of the names of the features
    :param feature_index: dict(str : int) linking the name of the features to their index
    :param threshold: threshold for the correlation coefficient

    :return: data, features, feature_index updated after dropping the feature(s) specified in feature_to_drop
    """
    nan_mask = np.isnan(data).any(axis=1)
    clean_data = data[~nan_mask]

    # Calculate the correlation matrix
    corr_matrix = np.corrcoef(clean_data, rowvar=False)
    num_cols = clean_data.shape[1]
    columns_to_remove = set()

    # Iterate through one triangle of the correlation matrix
    for i in range(num_cols):
        for j in range(i + 1, num_cols):
            correlation = corr_matrix[i, j]
            if abs(correlation) > threshold:
                # Decide which column to remove based on variance:
                # choosing the column with the highest variance preserves more
                # information and reduces data loss.
                var_i = np.var(clean_data[:, i])
                var_j = np.var(clean_data[:, j])
                if var_i > var_j:
                    columns_to_remove.add(j)
                else:
                    columns_to_remove.add(i)

    return drop_features(
        data, [features[i] for i in columns_to_remove], features, feature_index
    )
