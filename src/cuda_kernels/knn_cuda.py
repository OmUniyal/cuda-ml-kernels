"""
GPU-accelerated KNN operations using PyTorch.
"""

import torch
import numpy as np


def compute_distances_gpu(X_train, X_test):
    """
    Compute pairwise distances between test and train points.
    
    Parameters:
    -----------
    X_train : torch.Tensor, shape (n_train, n_features)
    X_test : torch.Tensor, shape (n_test, n_features)
    
    Returns:
    --------
    distances : torch.Tensor, shape (n_test, n_train)
    """
    # Efficient pairwise distance using cdist
    distances = torch.cdist(X_test, X_train, p=2)
    return distances


def find_k_nearest_neighbors(distances, k):
    """
    Find indices of k nearest neighbors for each test point.
    
    Parameters:
    -----------
    distances : torch.Tensor, shape (n_test, n_train)
    k : int
    
    Returns:
    --------
    neighbor_indices : torch.Tensor, shape (n_test, k)
    """
    # Get top-k smallest distances
    _, indices = torch.topk(distances, k, largest=False, dim=1)
    return indices


def knn_classify_gpu(X_train, y_train, X_test, k):
    """
    KNN classification on GPU.
    
    Parameters:
    -----------
    X_train : ndarray, shape (n_train, n_features)
    y_train : ndarray, shape (n_train,)
    X_test : ndarray, shape (n_test, n_features)
    k : int
    
    Returns:
    --------
    predictions : ndarray, shape (n_test,)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Move to GPU
    X_train_t = torch.from_numpy(X_train).to(device)
    y_train_t = torch.from_numpy(y_train).to(device)
    X_test_t = torch.from_numpy(X_test).to(device)
    
    # Compute distances
    distances = compute_distances_gpu(X_train_t, X_test_t)
    
    # Find k nearest neighbors
    neighbor_indices = find_k_nearest_neighbors(distances, k)
    
    # Gather labels of neighbors
    neighbor_labels = y_train_t[neighbor_indices]  # shape (n_test, k)
    
    # Vote: most common label
    predictions = torch.mode(neighbor_labels, dim=1).values
    
    return predictions.cpu().numpy()


def knn_regress_gpu(X_train, y_train, X_test, k):
    """
    KNN regression on GPU.
    
    Parameters:
    -----------
    X_train : ndarray, shape (n_train, n_features)
    y_train : ndarray, shape (n_train,)
    X_test : ndarray, shape (n_test, n_features)
    k : int
    
    Returns:
    --------
    predictions : ndarray, shape (n_test,)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Move to GPU
    X_train_t = torch.from_numpy(X_train).to(device)
    y_train_t = torch.from_numpy(y_train).to(device)
    X_test_t = torch.from_numpy(X_test).to(device)
    
    # Compute distances
    distances = compute_distances_gpu(X_train_t, X_test_t)
    
    # Find k nearest neighbors
    neighbor_indices = find_k_nearest_neighbors(distances, k)
    
    # Average neighbor values
    neighbor_values = y_train_t[neighbor_indices]  # shape (n_test, k)
    predictions = neighbor_values.mean(dim=1)
    
    return predictions.cpu().numpy()