"""
GPU-accelerated PCA using randomized SVD via PyTorch.
Optimized to keep data on GPU across operations.
"""

import torch
import numpy as np


def randomized_svd_gpu(X, n_components, n_iter=5, random_state=None):
    """
    Randomized SVD on GPU using PyTorch.

    Parameters:
    -----------
    X : torch.Tensor, shape (n_samples, n_features)
        Data matrix on GPU
    n_components : int
        Number of singular values/vectors to compute
    n_iter : int
        Number of power iterations for accuracy
    random_state : int or None

    Returns:
    --------
    U : torch.Tensor — left singular vectors (n_samples, n_components)
    S : torch.Tensor — singular values (n_components,)
    Vt : torch.Tensor — right singular vectors (n_components, n_features)
    """
    n_samples, n_features = X.shape
    n_random = n_components + 10  # oversample

    # Random projection matrix on GPU
    if random_state is not None:
        torch.manual_seed(random_state)
    Omega = torch.randn(n_features, n_random, device=X.device, dtype=X.dtype)

    # Power iterations for accuracy
    Y = X @ Omega
    for _ in range(n_iter):
        Y = X @ (X.T @ Y)
    Q, _ = torch.linalg.qr(Y)

    # Project and compute SVD on smaller matrix
    B = Q.T @ X
    U_b, S, Vt = torch.linalg.svd(B, full_matrices=False)

    # Recover U
    U = Q @ U_b

    # Truncate to requested components
    U = U[:, :n_components]
    S = S[:n_components]
    Vt = Vt[:n_components, :]

    return U, S, Vt


def pca_fit_gpu(X_np, n_components, scale=False, random_state=None):
    """
    Fit PCA on GPU and return components, explained variance, and mean.

    Parameters:
    -----------
    X_np : ndarray, shape (n_samples, n_features)
    n_components : int
    scale : bool — whether to standardize before PCA
    random_state : int or None

    Returns:
    --------
    dict with 'components', 'explained_variance', 'explained_variance_ratio',
               'singular_values', 'mean', 'std'
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = torch.from_numpy(X_np).to(device)

    # Center data
    mean = X.mean(dim=0)
    X_centered = X - mean

    # Optional scaling
    std = None
    if scale:
        std = X_centered.std(dim=0)
        std[std == 0] = 1.0  # avoid div by zero
        X_centered = X_centered / std

    # Randomized SVD
    U, S, Vt = randomized_svd_gpu(X_centered, n_components, random_state=random_state)

    # Explained variance
    n_samples = X.shape[0]
    explained_variance = (S ** 2) / (n_samples - 1)
    total_var = torch.sum(torch.var(X_centered, dim=0))
    explained_variance_ratio = explained_variance / total_var

    return {
        'components': Vt.cpu().numpy(),  # shape (n_components, n_features)
        'explained_variance': explained_variance.cpu().numpy(),
        'explained_variance_ratio': explained_variance_ratio.cpu().numpy(),
        'singular_values': S.cpu().numpy(),
        'mean': mean.cpu().numpy(),
        'std': std.cpu().numpy() if std is not None else None,
    }


def pca_transform_gpu(X_np, components, mean, std=None):
    """
    Transform new data using fitted PCA parameters.

    Parameters:
    -----------
    X_np : ndarray, shape (n_samples, n_features)
    components : ndarray, shape (n_components, n_features)
    mean : ndarray, shape (n_features,)
    std : ndarray or None

    Returns:
    --------
    X_transformed : ndarray, shape (n_samples, n_components)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = torch.from_numpy(X_np).to(device)
    comp = torch.from_numpy(components).to(device)
    mn = torch.from_numpy(mean).to(device)

    X_centered = X - mn
    if std is not None:
        sd = torch.from_numpy(std).to(device)
        X_centered = X_centered / sd

    return (X_centered @ comp.T).cpu().numpy()