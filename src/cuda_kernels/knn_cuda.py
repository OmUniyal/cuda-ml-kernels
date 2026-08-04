"""
GPU-accelerated KNN operations using PyTorch.
Smart dispatch: uses GPU only when beneficial, falls back to CPU otherwise.
Uses float32 for 2x memory bandwidth savings on GPU.
"""

import torch
import numpy as np
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

# Threshold: GPU only wins above this many total distance computations
# (n_train * n_test). Below this, CPU sklearn is faster due to zero transfer overhead.
_GPU_MIN_OPS = 25_000_000  # ~50K x 5K or 100K x 250 test split


def _should_use_gpu(n_train, n_test):
    """Heuristic: GPU is worth it only for large enough problems."""
    if not torch.cuda.is_available():
        return False
    total_ops = n_train * n_test
    return total_ops > _GPU_MIN_OPS


def compute_distances_gpu(X_train, X_test, chunk_size=2048):
    """
    Compute pairwise distances on GPU with float32 for speed.
    Processes in fixed-size chunks.
    """
    n_test = X_test.shape[0]

    if n_test <= chunk_size:
        return torch.cdist(X_test, X_train, p=2)

    distances_list = []
    for i in range(0, n_test, chunk_size):
        end = min(i + chunk_size, n_test)
        chunk = X_test[i:end]
        dist_chunk = torch.cdist(chunk, X_train, p=2)
        distances_list.append(dist_chunk)

    return torch.cat(distances_list, dim=0)


def knn_classify_gpu(X_train, y_train, X_test, k, chunk_size=2048):
    """
    KNN classification — auto-selects GPU or CPU based on problem size.
    """
    n_train, n_test = X_train.shape[0], X_test.shape[0]

    # CPU fallback for small problems (sklearn is faster)
    if not _should_use_gpu(n_train, n_test):
        clf = KNeighborsClassifier(n_neighbors=min(k, n_train))
        clf.fit(X_train, y_train)
        return clf.predict(X_test)

    device = torch.device("cuda")

    # Use float32 for 2x memory bandwidth, convert back at the end
    X_train_t = torch.from_numpy(X_train).float().to(device)
    y_train_t = torch.from_numpy(y_train).to(device)
    X_test_t = torch.from_numpy(X_test).float().to(device)

    distances = compute_distances_gpu(X_train_t, X_test_t, chunk_size=chunk_size)

    # topk on GPU
    k_eff = min(k, n_train)
    _, neighbor_indices = torch.topk(distances, k_eff, largest=False, dim=1)

    neighbor_labels = y_train_t[neighbor_indices]
    predictions = torch.mode(neighbor_labels, dim=1).values

    return predictions.cpu().numpy()


def knn_regress_gpu(X_train, y_train, X_test, k, chunk_size=2048):
    """
    KNN regression — auto-selects GPU or CPU based on problem size.
    """
    n_train, n_test = X_train.shape[0], X_test.shape[0]

    if not _should_use_gpu(n_train, n_test):
        reg = KNeighborsRegressor(n_neighbors=min(k, n_train))
        reg.fit(X_train, y_train)
        return reg.predict(X_test)

    device = torch.device("cuda")

    X_train_t = torch.from_numpy(X_train).float().to(device)
    y_train_t = torch.from_numpy(y_train).float().to(device)
    X_test_t = torch.from_numpy(X_test).float().to(device)

    distances = compute_distances_gpu(X_train_t, X_test_t, chunk_size=chunk_size)

    k_eff = min(k, n_train)
    _, neighbor_indices = torch.topk(distances, k_eff, largest=False, dim=1)

    neighbor_values = y_train_t[neighbor_indices]
    predictions = neighbor_values.mean(dim=1)

    return predictions.cpu().numpy()