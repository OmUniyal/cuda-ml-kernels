"""
GPU-accelerated K-Means operations using PyTorch.
Optimized to minimize CPU-GPU transfers.
"""

import torch
import numpy as np


class KMeansGPU:
    """
    GPU-optimized K-Means that keeps data on GPU across iterations.
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
    
    def fit(self, X_np):
        """
        Fit K-Means to data.
        
        Parameters:
        -----------
        X_np : numpy array, shape (n_samples, n_features)
        """
        # Move data to GPU ONCE and keep it there
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X = torch.from_numpy(X_np).to(device)
        n_samples, n_features = X.shape
        
        # Initialize centroids on GPU
        rng = np.random.RandomState(self.random_state)
        indices = rng.choice(n_samples, self.n_clusters, replace=False)
        centroids = X[indices].clone()
        
        prev_inertia = float('inf')
        
        for i in range(self.max_iter):
            # E-step: compute distances and assign labels (all on GPU)
            distances = torch.cdist(X, centroids, p=2) ** 2
            labels = torch.argmin(distances, dim=1)
            
            # Compute inertia on GPU
            inertia = torch.sum(torch.min(distances, dim=1)[0]).item()
            
            if self.verbose and i % 10 == 0:
                print(f"Iteration {i}: inertia = {inertia:.4f}")
            
            # Check convergence
            if abs(prev_inertia - inertia) < self.tol:
                if self.verbose:
                    print(f"Converged at iteration {i}")
                break
            
            prev_inertia = inertia
            
            # M-step: recompute centroids (vectorized on GPU)
            new_centroids = torch.zeros_like(centroids)
            counts = torch.zeros(self.n_clusters, device=device, dtype=torch.float64)
            
            for c in range(self.n_clusters):
                mask = (labels == c)
                count = mask.sum().item()
                if count > 0:
                    new_centroids[c] = X[mask].mean(dim=0)
                else:
                    # Reinitialize empty cluster
                    random_idx = torch.randint(0, n_samples, (1,), device=device)
                    new_centroids[c] = X[random_idx].squeeze()
            
            centroids = new_centroids
            self.n_iter = i + 1
        
        # Copy results back to CPU only at the end
        self.centroids = centroids.cpu().numpy()
        self.labels = labels.cpu().numpy()
        self.inertia = inertia
        
        return self
    
    def predict(self, X_np):
        """
        Predict closest cluster for each sample.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X = torch.from_numpy(X_np).to(device)
        centroids = torch.from_numpy(self.centroids).to(device)
        
        distances = torch.cdist(X, centroids, p=2) ** 2
        labels = torch.argmin(distances, dim=1)
        
        return labels.cpu().numpy()