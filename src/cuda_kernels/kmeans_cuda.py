"""
GPU-accelerated K-Means operations using PyTorch.
"""

import torch
import numpy as np


def compute_distances_gpu(data, centroids):
    """
    Compute squared Euclidean distances between all points and centroids on GPU.
    
    Parameters:
    -----------
    data : torch.Tensor, shape (n_samples, n_features) — on GPU
    centroids : torch.Tensor, shape (n_clusters, n_features) — on GPU
    
    Returns:
    --------
    distances : torch.Tensor, shape (n_samples, n_clusters) — on GPU
    labels : torch.Tensor, shape (n_samples,) — on GPU
    """
    # torch.cdist computes pairwise distances efficiently on GPU
    distances = torch.cdist(data, centroids, p=2)  # p=2 = Euclidean
    distances = distances ** 2  # squared Euclidean
    
    # Assign each point to nearest centroid
    labels = torch.argmin(distances, dim=1)
    
    return distances, labels


def update_centroids_gpu(data, labels, n_clusters):
    """
    Compute new centroids as mean of assigned points on GPU.
    
    Parameters:
    -----------
    data : torch.Tensor, shape (n_samples, n_features) — on GPU
    labels : torch.Tensor, shape (n_samples,) — on GPU
    n_clusters : int
    
    Returns:
    --------
    centroids : torch.Tensor, shape (n_clusters, n_features) — on GPU
    """
    n_features = data.shape[1]
    centroids = torch.zeros((n_clusters, n_features), device=data.device, dtype=data.dtype)
    
    for c in range(n_clusters):
        mask = (labels == c)
        if mask.sum() > 0:
            centroids[c] = data[mask].mean(dim=0)
        else:
            # Reinitialize empty cluster to random point
            random_idx = torch.randint(0, len(data), (1,), device=data.device)
            centroids[c] = data[random_idx].squeeze()
    
    return centroids


def launch_distance_kernel(data_np, centroids_np):
    """
    Helper: numpy in, numpy out. Handles GPU transfer internally.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    data = torch.from_numpy(data_np).to(device)
    centroids = torch.from_numpy(centroids_np).to(device)
    
    distances, labels = compute_distances_gpu(data, centroids)
    
    return labels.cpu().numpy(), distances.cpu().numpy()


def launch_centroid_update_kernel(data_np, labels_np, n_clusters, n_features):
    """
    Helper: numpy in, numpy out. Handles GPU transfer internally.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    data = torch.from_numpy(data_np).to(device)
    labels = torch.from_numpy(labels_np).to(device)
    
    centroids = update_centroids_gpu(data, labels, n_clusters)
    
    return centroids.cpu().numpy()