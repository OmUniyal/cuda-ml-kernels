"""
CUDA kernels for K-Means clustering using Numba.
"""

import numpy as np
from numba import cuda
import math


@cuda.jit
def compute_distances_kernel(data, centroids, distances, labels):
    """
    CUDA kernel: For each point, compute squared Euclidean distance to all centroids.
    Assign point to nearest centroid.
    
    Parameters:
    -----------
    data : device array, shape (n_samples, n_features)
    centroids : device array, shape (n_clusters, n_features)
    distances : device array, shape (n_samples, n_clusters) — output
    labels : device array, shape (n_samples,) — output cluster assignments
    """
    idx = cuda.grid(1)
    n_samples = data.shape[0]
    n_clusters = centroids.shape[0]
    n_features = data.shape[1]
    
    if idx >= n_samples:
        return
    
    min_dist = math.inf
    best_label = 0
    
    for c in range(n_clusters):
        dist = 0.0
        for f in range(n_features):
            diff = data[idx, f] - centroids[c, f]
            dist += diff * diff
        
        distances[idx, c] = dist
        
        if dist < min_dist:
            min_dist = dist
            best_label = c
    
    labels[idx] = best_label


@cuda.jit
def compute_new_centroids_kernel(data, labels, centroids, counts):
    """
    CUDA kernel: Compute new centroids as mean of assigned points.
    Uses parallel reduction pattern.
    
    Parameters:
    -----------
    data : device array, shape (n_samples, n_features)
    labels : device array, shape (n_samples,)
    centroids : device array, shape (n_clusters, n_features) — output
    counts : device array, shape (n_clusters,) — output point counts per cluster
    """
    idx = cuda.grid(1)
    n_samples = data.shape[0]
    
    if idx >= n_samples:
        return
    
    cluster = labels[idx]
    
    # Atomically accumulate
    for f in range(data.shape[1]):
        cuda.atomic.add(centroids, (cluster, f), data[idx, f])
    
    cuda.atomic.add(counts, cluster, 1)


def launch_distance_kernel(data, centroids):
    """
    Helper to launch the distance computation kernel.
    
    Returns labels and distances on host.
    """
    n_samples = data.shape[0]
    n_clusters = centroids.shape[0]
    
    # Transfer to device
    d_data = cuda.to_device(data)
    d_centroids = cuda.to_device(centroids)
    d_distances = cuda.device_array((n_samples, n_clusters), dtype=np.float64)
    d_labels = cuda.device_array(n_samples, dtype=np.int32)
    
    # Configure grid
    threads_per_block = 256
    blocks_per_grid = (n_samples + threads_per_block - 1) // threads_per_block
    
    # Launch kernel
    compute_distances_kernel[blocks_per_grid, threads_per_block](
        d_data, d_centroids, d_distances, d_labels
    )
    
    # Copy back
    labels = d_labels.copy_to_host()
    distances = d_distances.copy_to_host()
    
    return labels, distances


def launch_centroid_update_kernel(data, labels, n_clusters, n_features):
    """
    Helper to launch the centroid update kernel.
    
    Returns new centroids on host.
    """
    n_samples = data.shape[0]
    
    # Transfer to device
    d_data = cuda.to_device(data)
    d_labels = cuda.to_device(labels)
    d_centroids = cuda.device_array((n_clusters, n_features), dtype=np.float64)
    d_counts = cuda.device_array(n_clusters, dtype=np.int32)
    
    # Zero out arrays
    d_centroids.copy_to_host()  # This is a hack — better to zero on device
    # Actually let's zero properly
    centroids_host = np.zeros((n_clusters, n_features), dtype=np.float64)
    counts_host = np.zeros(n_clusters, dtype=np.int32)
    d_centroids = cuda.to_device(centroids_host)
    d_counts = cuda.to_device(counts_host)
    
    # Configure grid
    threads_per_block = 256
    blocks_per_grid = (n_samples + threads_per_block - 1) // threads_per_block
    
    # Launch kernel
    compute_new_centroids_kernel[blocks_per_grid, threads_per_block](
        d_data, d_labels, d_centroids, d_counts
    )
    
    # Copy back and normalize
    centroids = d_centroids.copy_to_host()
    counts = d_counts.copy_to_host()
    
    # Avoid division by zero
    for c in range(n_clusters):
        if counts[c] > 0:
            centroids[c] /= counts[c]
        else:
            # Reinitialize empty cluster to random point
            centroids[c] = data[np.random.randint(n_samples)]
    
    return centroids