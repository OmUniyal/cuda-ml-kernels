"""
Unit tests for GPU-accelerated PCA.
Validated against scikit-learn PCA.
"""

import numpy as np
import pytest
from sklearn.decomposition import PCA as SklearnPCA
from algorithms.pca import PCA


class TestPCA:
    @pytest.fixture
    def data(self):
        """Generate high-dimensional data."""
        np.random.seed(42)
        return np.random.randn(1000, 50)

    def test_transform_shape(self, data):
        """Transformed data should have correct shape."""
        pca = PCA(n_components=10)
        X_new = pca.fit_transform(data)
        assert X_new.shape == (1000, 10)

    def test_explained_variance_sum(self, data):
        """Explained variance ratios should sum to <= 1."""
        pca = PCA(n_components=10)
        pca.fit(data)
        assert np.sum(pca.explained_variance_ratio_) <= 1.0 + 1e-6
        assert np.all(pca.explained_variance_ratio_ >= 0)

    def test_components_orthonormal(self, data):
        """Components should be orthonormal."""
        pca = PCA(n_components=10)
        pca.fit(data)
        dot = pca.components_ @ pca.components_.T
        np.testing.assert_allclose(dot, np.eye(10), atol=1e-5)

    def test_inverse_transform_shape(self, data):
        """Inverse transform should restore original shape."""
        pca = PCA(n_components=10)
        X_new = pca.fit_transform(data)
        X_rec = pca.inverse_transform(X_new)
        assert X_rec.shape == data.shape

    def test_reconstruction_error(self, data):
        """Reconstruction error should decrease with more components."""
        pca_5 = PCA(n_components=5).fit(data)
        pca_20 = PCA(n_components=20).fit(data)

        err_5 = np.mean((data - pca_5.inverse_transform(pca_5.transform(data))) ** 2)
        err_20 = np.mean((data - pca_20.inverse_transform(pca_20.transform(data))) ** 2)

        assert err_20 < err_5

    def test_matches_sklearn(self, data):
        """Explained variance ratio should match sklearn closely."""
        our_pca = PCA(n_components=10, random_state=42)
        our_pca.fit(data)

        sk_pca = SklearnPCA(n_components=10, svd_solver='randomized', random_state=42)
        sk_pca.fit(data)

        # Explained variance ratio should be close (randomized SVD has some variance)
        np.testing.assert_allclose(
            our_pca.explained_variance_ratio_,
            sk_pca.explained_variance_ratio_,
            rtol=0.15, atol=0.01
        )

    def test_float_n_components(self, data):
        """Float n_components should select by cumulative variance."""
        pca = PCA(n_components=0.95)
        X_new = pca.fit_transform(data)
        cumsum = np.sum(pca.explained_variance_ratio_)
        assert cumsum >= 0.95

    def test_scaled_pca(self, data):
        """Scaled PCA should produce different but valid results."""
        pca = PCA(n_components=5, scale=True)
        X_new = pca.fit_transform(data)
        assert X_new.shape == (1000, 5)
        assert np.all(np.isfinite(X_new))


class TestPCAEdgeCases:
    def test_n_components_larger_than_features(self):
        """Should cap at min(n_samples, n_features)."""
        X = np.random.randn(10, 3)
        pca = PCA(n_components=10)
        X_new = pca.fit_transform(X)
        assert X_new.shape[1] == 3

    def test_single_feature(self):
        """Should handle single feature gracefully."""
        X = np.random.randn(100, 1)
        pca = PCA(n_components=1)
        X_new = pca.fit_transform(X)
        assert X_new.shape == (100, 1)