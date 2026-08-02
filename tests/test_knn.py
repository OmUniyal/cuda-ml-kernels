"""
Unit tests for KNN classifier and regressor.
Validated against scikit-learn KNeighborsClassifier/Regressor.
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from algorithms.knn import KNN


class TestKNNClassification:
    @pytest.fixture
    def clf_data(self):
        """Generate classification dataset."""
        X, y = make_classification(
            n_samples=1000, n_features=10, n_classes=3,
            n_informative=5, random_state=42
        )
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def test_predict_shape(self, clf_data):
        """Predictions should match test set size."""
        X_train, X_test, y_train, y_test = clf_data
        knn = KNN(n_neighbors=5, task='classification')
        knn.fit(X_train, y_train)
        preds = knn.predict(X_test)
        assert preds.shape == y_test.shape

    def test_accuracy_matches_sklearn(self, clf_data):
        """GPU KNN accuracy should be close to scikit-learn."""
        X_train, X_test, y_train, y_test = clf_data

        # Our GPU implementation
        knn = KNN(n_neighbors=5, task='classification')
        knn.fit(X_train, y_train)
        our_acc = knn.score(X_test, y_test)

        # scikit-learn baseline
        sk_knn = KNeighborsClassifier(n_neighbors=5)
        sk_knn.fit(X_train, y_train)
        sk_acc = sk_knn.score(X_test, y_test)

        # Allow small tolerance due to tie-breaking differences
        assert abs(our_acc - sk_acc) < 0.05, (
            f"Accuracy mismatch: ours={our_acc:.4f}, sklearn={sk_acc:.4f}"
        )

    def test_different_k_values(self, clf_data):
        """Should work with different k values."""
        X_train, X_test, y_train, y_test = clf_data
        for k in [1, 3, 5, 10]:
            knn = KNN(n_neighbors=k, task='classification')
            knn.fit(X_train, y_train)
            preds = knn.predict(X_test)
            assert len(preds) == len(y_test)
            assert len(np.unique(preds)) >= 1

    def test_k_larger_than_train(self):
        """k larger than training size should still work (clamped)."""
        X_train = np.random.randn(5, 3)
        y_train = np.array([0, 1, 0, 1, 0])
        X_test = np.random.randn(2, 3)

        knn = KNN(n_neighbors=10, task='classification')
        knn.fit(X_train, y_train)
        preds = knn.predict(X_test)
        assert preds.shape == (2,)


class TestKNNRegression:
    @pytest.fixture
    def reg_data(self):
        """Generate regression dataset."""
        X, y = make_regression(
            n_samples=1000, n_features=10, noise=10, random_state=42
        )
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def test_predict_shape(self, reg_data):
        """Predictions should match test set size."""
        X_train, X_test, y_train, y_test = reg_data
        knn = KNN(n_neighbors=5, task='regression')
        knn.fit(X_train, y_train)
        preds = knn.predict(X_test)
        assert preds.shape == y_test.shape
        assert preds.dtype == np.float64 or preds.dtype == np.float32

    def test_r2_matches_sklearn(self, reg_data):
        """GPU KNN R² should be close to scikit-learn."""
        X_train, X_test, y_train, y_test = reg_data

        knn = KNN(n_neighbors=5, task='regression')
        knn.fit(X_train, y_train)
        our_r2 = knn.score(X_test, y_test)

        sk_knn = KNeighborsRegressor(n_neighbors=5)
        sk_knn.fit(X_train, y_train)
        sk_r2 = sk_knn.score(X_test, y_test)

        assert abs(our_r2 - sk_r2) < 0.05, (
            f"R² mismatch: ours={our_r2:.4f}, sklearn={sk_r2:.4f}"
        )

    def test_regression_output_range(self, reg_data):
        """Regression predictions should be within training target range."""
        X_train, X_test, y_train, y_test = reg_data
        knn = KNN(n_neighbors=5, task='regression')
        knn.fit(X_train, y_train)
        preds = knn.predict(X_test)

        assert preds.min() >= y_train.min() - 1e-6
        assert preds.max() <= y_train.max() + 1e-6


class TestKNNEdgeCases:
    def test_single_feature(self):
        """KNN should work with single feature."""
        X_train = np.array([[1], [2], [3], [4], [5]])
        y_train = np.array([0, 0, 1, 1, 1])
        X_test = np.array([[2.5], [4.5]])

        knn = KNN(n_neighbors=3, task='classification')
        knn.fit(X_train, y_train)
        preds = knn.predict(X_test)
        assert preds.shape == (2,)

    def test_binary_classification(self):
        """Binary classification should produce only 2 classes."""
        X, y = make_classification(
            n_samples=500, n_features=5, n_classes=2,
            n_informative=3, random_state=42
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        knn = KNN(n_neighbors=5, task='classification')
        knn.fit(X_train, y_train)
        preds = knn.predict(X_test)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_large_k(self):
        """Large k should not crash."""
        X_train = np.random.randn(100, 5)
        y_train = np.random.randint(0, 3, 100)
        X_test = np.random.randn(10, 5)

        knn = KNN(n_neighbors=50, task='classification')
        knn.fit(X_train, y_train)
        preds = knn.predict(X_test)
        assert len(preds) == 10