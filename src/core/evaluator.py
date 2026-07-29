"""
Evaluation metrics for clustering algorithms.
"""

import numpy as np
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score


class ClusteringEvaluator:
    """
    Compute clustering quality metrics.
    """
    
    @staticmethod
    def inertia(X, labels, centroids):
        """
        Sum of squared distances to nearest centroid.
        
        Parameters:
        -----------
        X : ndarray, shape (n_samples, n_features)
        labels : ndarray, shape (n_samples,)
        centroids : ndarray, shape (n_clusters, n_features)
        
        Returns:
        --------
        inertia : float
        """
        total = 0.0
        for c in range(len(centroids)):
            mask = (labels == c)
            if mask.sum() > 0:
                diff = X[mask] - centroids[c]
                total += np.sum(diff ** 2)
        return total
    
    @staticmethod
    def silhouette(X, labels):
        """
        Silhouette score: -1 (bad) to 1 (good).
        
        Requires at least 2 clusters and 2 samples per cluster.
        """
        n_clusters = len(np.unique(labels))
        if n_clusters < 2:
            return None
        try:
            return silhouette_score(X, labels)
        except ValueError:
            return None
    
    @staticmethod
    def adjusted_rand(true_labels, pred_labels):
        """
        Adjusted Rand Index: -1 (random) to 1 (perfect).
        """
        return adjusted_rand_score(true_labels, pred_labels)
    
    @staticmethod
    def nmi(true_labels, pred_labels):
        """
        Normalized Mutual Information: 0 (none) to 1 (perfect).
        """
        return normalized_mutual_info_score(true_labels, pred_labels)
    
    @classmethod
    def evaluate(cls, X, labels, centroids=None, true_labels=None):
        """
        Compute all available metrics.
        
        Returns:
        --------
        dict of metric names to values
        """
        results = {}
        
        if centroids is not None:
            results['inertia'] = cls.inertia(X, labels, centroids)
        
        sil = cls.silhouette(X, labels)
        if sil is not None:
            results['silhouette'] = sil
        
        if true_labels is not None:
            results['adjusted_rand_index'] = cls.adjusted_rand(true_labels, labels)
            results['nmi'] = cls.nmi(true_labels, labels)
        
        return results