"""
GPU-accelerated K-Means clustering using PyTorch.
"""

from cuda_kernels.kmeans_cuda import KMeansGPU


class KMeans(KMeansGPU):
    """
    Scikit-learn compatible API for GPU K-Means.
    """
    pass