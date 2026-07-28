"""
GPU-accelerated K-Means clustering using Numba CUDA.
"""

import numpy as np
from cuda_kernels.kmeans_cuda import launch_distance_kernel, launch_centroid_update_kernel


class KMeans:
    """
    K-Means clustering with CUDA acceleration.
    
    Parameters:
    -----------
    n_clusters : int, default=8
        Number of clusters.
    max_iter : int, default=300
        Maximum iterations.
    tol : float, default=1e-4
        Convergence tolerance.
    random_state : int, optional
        Random seed for centroid initialization.
    verbose : bool, default=False
        Print progress.
    """
    
    def __init__(self, n_clusters=8, max_iter=300, tol=1e-4, random_state=None, verbose=False):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose
        
        self.centroids = None
        self.labels = None
        self.inertia = None
        self.n_iter = 0
    
    def fit(self, X):
        """
        Fit K-Means to data.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
        
        Returns:
        --------
        self
        """
        X = np.asarray(X, dtype=np.float64)
        n_samples, n_features = X.shape
        
        # Initialize centroids
        rng = np.random.RandomState(self.random_state)
        indices = rng.choice(n_samples, self.n_clusters, replace=False)
        self.centroids = X[indices].copy()
        
        prev_inertia = np.inf
        
        for i in range(self.max_iter):
            # E-step: assign points to nearest centroid (GPU)
            self.labels, distances = launch_distance_kernel(X, self.centroids)
            
            # Compute inertia (sum of min distances)
            self.inertia = np.sum(np.min(distances, axis=1))
            
            if self.verbose and i % 10 == 0:
                print(f"Iteration {i}: inertia = {self.inertia:.4f}")
            
            # Check convergence
            if abs(prev_inertia - self.inertia) < self.tol:
                if self.verbose:
                    print(f"Converged at iteration {i}")
                break
            
            prev_inertia = self.inertia
            
            # M-step: recompute centroids (GPU)
            self.centroids = launch_centroid_update_kernel(
                X, self.labels, self.n_clusters, n_features
            )
            
            self.n_iter = i + 1
        
        return self
    
    def predict(self, X):
        """
        Predict closest cluster for each sample.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
        
        Returns:
        --------
        labels : array, shape (n_samples,)
        """
        X = np.asarray(X, dtype=np.float64)
        labels, _ = launch_distance_kernel(X, self.centroids)
        return labels
    
    def fit_predict(self, X):
        """
        Fit and predict.
        """
        self.fit(X)
        return self.labels