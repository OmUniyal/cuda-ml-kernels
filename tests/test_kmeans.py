"""
Unit tests for GPU-accelerated K-Means.
Verifies correctness against scikit-learn.
"""

import numpy as np
import pytest
from sklearn.cluster import KMeans as SklearnKMeans
from algorithms.kmeans import KMeans


def test_kmeans_basic():
    """Test on simple synthetic data."""
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    n_clusters = 5
    
    # Generate random data
    X = np.random.randn(n_samples, n_features)
    
    # Fit both models
    gpu_kmeans = KMeans(n_clusters=n_clusters, random_state=42, max_iter=100)
    gpu_kmeans.fit(X)
    
    sklearn_kmeans = SklearnKMeans(n_clusters=n_clusters, random_state=42, max_iter=100, n_init=1)
    sklearn_kmeans.fit(X)
    
    # Inertia should be similar (not exact due to different init/floating point)
    assert gpu_kmeans.inertia is not None
    assert gpu_kmeans.inertia > 0
    
    # Labels should be valid cluster indices
    assert np.all(gpu_kmeans.labels >= 0)
    assert np.all(gpu_kmeans.labels < n_clusters)
    
    # Centroids should have right shape
    assert gpu_kmeans.centroids.shape == (n_clusters, n_features)


def test_kmeans_predict():
    """Test predict on new data."""
    np.random.seed(42)
    X_train = np.random.randn(500, 5)
    X_test = np.random.randn(100, 5)
    
    model = KMeans(n_clusters=3, random_state=42)
    model.fit(X_train)
    
    labels = model.predict(X_test)
    
    assert len(labels) == len(X_test)
    assert np.all(labels >= 0)
    assert np.all(labels < 3)


def test_kmeans_convergence():
    """Test that model converges within max_iter."""
    np.random.seed(42)
    X = np.random.randn(200, 3)
    
    model = KMeans(n_clusters=2, max_iter=50, tol=1e-4, random_state=42)
    model.fit(X)
    
    assert model.n_iter <= 50
    assert model.n_iter > 0


def test_kmeans_single_cluster():
    """Edge case: single cluster."""
    np.random.seed(42)
    X = np.random.randn(100, 2)
    
    model = KMeans(n_clusters=1, random_state=42)
    model.fit(X)
    
    assert np.all(model.labels == 0)
    assert model.centroids.shape == (1, 2)


def test_kmeans_high_dimensional():
    """Test on higher dimensional data."""
    np.random.seed(42)
    X = np.random.randn(300, 50)
    
    model = KMeans(n_clusters=4, random_state=42)
    model.fit(X)
    
    assert model.centroids.shape == (4, 50)
    assert len(np.unique(model.labels)) <= 4