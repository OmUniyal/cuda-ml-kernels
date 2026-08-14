"""
GPU-accelerated Principal Component Analysis using PyTorch.
scikit-learn compatible API.
"""

import numpy as np
from cuda_kernels.pca_cuda import pca_fit_gpu, pca_transform_gpu


class PCA:
    """
    Principal Component Analysis with GPU acceleration.

    Uses randomized SVD for efficiency on high-dimensional data.

    Parameters:
    -----------
    n_components : int or float or 'mle'
        Number of components to keep. If float, select by explained variance ratio.
    scale : bool, default=False
        Whether to standardize features before PCA.
    random_state : int or None, default=None
    """

    def __init__(self, n_components=2, scale=False, random_state=None):
        self.n_components = n_components
        self.scale = scale
        self.random_state = random_state

        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.singular_values_ = None
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        """
        Fit PCA to data.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)

        Returns:
        --------
        self
        """
        X = np.asarray(X, dtype=np.float64)
        n_samples, n_features = X.shape

        # Determine n_components
        if isinstance(self.n_components, float):
            # Will select after fitting
            n_comp = min(n_samples, n_features)
        else:
            n_comp = min(self.n_components, n_features, n_samples)

        result = pca_fit_gpu(
            X, n_components=n_comp,
            scale=self.scale,
            random_state=self.random_state
        )

        self.components_ = result['components']
        self.explained_variance_ = result['explained_variance']
        self.explained_variance_ratio_ = result['explained_variance_ratio']
        self.singular_values_ = result['singular_values']
        self.mean_ = result['mean']
        self.std_ = result['std']

        # If float n_components, select by cumulative variance
        if isinstance(self.n_components, float):
            cumsum = np.cumsum(self.explained_variance_ratio_)
            n_comp = np.searchsorted(cumsum, self.n_components) + 1
            self.components_ = self.components_[:n_comp]
            self.explained_variance_ = self.explained_variance_[:n_comp]
            self.explained_variance_ratio_ = self.explained_variance_ratio_[:n_comp]
            self.singular_values_ = self.singular_values_[:n_comp]

        return self

    def transform(self, X):
        """
        Apply dimensionality reduction.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)

        Returns:
        --------
        X_new : ndarray, shape (n_samples, n_components)
        """
        X = np.asarray(X, dtype=np.float64)
        return pca_transform_gpu(X, self.components_, self.mean_, self.std_)

    def fit_transform(self, X):
        """Fit and transform in one call."""
        return self.fit(X).transform(X)

    def inverse_transform(self, X_transformed):
        """
        Transform data back to original space.

        Parameters:
        -----------
        X_transformed : array-like, shape (n_samples, n_components)

        Returns:
        --------
        X_original : ndarray, shape (n_samples, n_features)
        """
        X_t = np.asarray(X_transformed, dtype=np.float64)
        X_approx = X_t @ self.components_
        if self.std_ is not None:
            X_approx = X_approx * self.std_
        return X_approx + self.mean_

    @property
    def n_components_(self):
        """Actual number of components after fitting."""
        return self.components_.shape[0] if self.components_ is not None else None