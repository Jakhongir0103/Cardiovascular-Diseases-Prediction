import numpy as np


class CustomPCA:
    def __init__(self):
        self.eigenvalues = None
        self.eigenvectors = None

    def fit(self, data):
        """
        Performs principal component analysis on the given data and saves the eigenvalues and eigenvectors.
        :param data: The data to perform PCA on.
        :param k: The number of principal components to use.
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
        sorted_eigenvectors = eigenvectors[:, order_of_importance]

        self.eigenvalues = sorted_eigenvalues
        self.eigenvectors = sorted_eigenvectors

    def transform(self, data: np.ndarray, k: int = None, threshold: float = 0.8):
        """
        Apply PCA to the given data using the previously saved eigenvalues and eigenvectors.
        If k is not specified, the number of principal components is chosen such that the
        cumulative explained variance is above the threshold.

        :param data: The data to reduce dimensionality on.
        :param threshold: The threshold for the cumulative explained variance.
        :param k: The number of principal components to use.
        :return: The reduced data.
        """
        if k is None:
            k = self._num_of_PCs(threshold)

        standardized_data = data - data.mean(axis=0)
        reduced_data = np.matmul(standardized_data, self.eigenvectors[:, :k])
        return reduced_data

    def explained_variance(self):
        """
        Return the explained variance of the saved eigenvalues.
        """
        return self.eigenvalues / np.sum(self.eigenvalues)

    def _num_of_PCs(self,
                    threshold: float) -> int:
        """
        Returns the number of principal components for which
        the cumulative explained variance is above the threshold.
        :param threshold: threshold for the cumulative explained variance
        :return: number of principal components
        """
        cumulative_explained_vars = 0
        explained_vars = self.explained_variance()

        for i in range(len(self.eigenvalues)):
            cumulative_explained_vars += explained_vars[i]
            if cumulative_explained_vars > threshold:
                return i + 1
        return len(self.eigenvalues)
