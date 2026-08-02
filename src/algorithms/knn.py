"""
GPU-accelerated K-Nearest Neighbors using PyTorch.
"""

import numpy as np
from cuda_kernels.knn_cuda import knn_classify_gpu, knn_regress_gpu


class KNN:
    """
    K-Nearest Neighbors classifier/regressor with GPU acceleration.
    
    Parameters:
    -----------
    n_neighbors : int, default=5
        Number of neighbors to use.
    task : str, default='classification'
        'classification' or 'regression'
    """
    
    def __init__(self, n_neighbors=5, task='classification'):
        self.n_neighbors = n_neighbors
        self.task = task
        
        self.X_train = None
        self.y_train = None
    
    def fit(self, X, y):
        """
        Store training data. KNN is a "lazy" learner.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
        y : array-like, shape (n_samples,)
        
        Returns:
        --------
        self
        """
        self.X_train = np.asarray(X, dtype=np.float64)
        self.y_train = np.asarray(y)
        return self
    
    def predict(self, X):
        """
        Predict labels/values for test data.
        
        Parameters:
        -----------
        X : array-like, shape (n_test, n_features)
        
        Returns:
        --------
        predictions : ndarray, shape (n_test,)
        """
        X_test = np.asarray(X, dtype=np.float64)
        
        if self.task == 'classification':
            return knn_classify_gpu(
                self.X_train, 
                self.y_train, 
                X_test, 
                self.n_neighbors
            )
        else:
            return knn_regress_gpu(
                self.X_train, 
                self.y_train, 
                X_test, 
                self.n_neighbors
            )
    
    def score(self, X, y):
        """
        Compute accuracy (classification) or R² (regression).
        """
        from sklearn.metrics import accuracy_score, r2_score
        
        predictions = self.predict(X)
        y_true = np.asarray(y)
        
        if self.task == 'classification':
            return accuracy_score(y_true, predictions)
        else:
            return r2_score(y_true, predictions)