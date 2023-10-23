import numpy as np


def custom_pca(data: np.ndarray, k: int, features: list[str]):
    """
    Performs principal component analysis on the given data and returns the reduced data and features.

    :param data: The data to perform PCA on.
    :param k: The number of principal components to use.
    :param features: The features of the data.
    :return: The reduced data and features.
    """
    # Standardize data
    standardized_data = data - data.mean(axis=0)

    # Compute covariance matrix
    covariance_matrix = np.cov(standardized_data, ddof=1, rowvar=False)

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    # Sort eigenvalues and eigenvectors
    order_of_importance = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[order_of_importance]
    sorted_eigenvectors = eigenvectors[:, order_of_importance]  # sort the columns
    sorted_features = [features[i] for i in order_of_importance]

    # use sorted_eigenvalues to ensure the explained variances correspond to the eigenvectors
    explained_variance = sorted_eigenvalues / np.sum(sorted_eigenvalues)

    # select the number of principal components
    reduced_data = np.matmul(
        standardized_data, sorted_eigenvectors[:, :k]
    )  # transform the original data
    reduced_features = sorted_features[:k]

    return reduced_data, reduced_features
