"""
Auto-preprocessing pipeline for clustering data.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder


class Preprocessor:
    """
    Preprocess data for clustering algorithms.
    
    Steps: scale → encode → reduce dimensions (optional)
    """
    
    SCALERS = {
        'standard': StandardScaler,
        'minmax': MinMaxScaler,
        'none': None
    }
    
    def __init__(self, scale='standard', reduce_dim=None, n_components=2):
        """
        Parameters:
        -----------
        scale : str — 'standard', 'minmax', or 'none'
        reduce_dim : str or None — 'pca' (planned) or None
        n_components : int — dimensions after reduction
        """
        self.scale = scale
        self.reduce_dim = reduce_dim
        self.n_components = n_components
        self.scaler = None
        self._fitted = False
    
    def fit_transform(self, X):
        """
        Fit and transform data.
        
        Returns:
        --------
        X_processed : ndarray
        """
        X = np.asarray(X, dtype=np.float64)
        
        # Handle scaling
        if self.scale != 'none':
            scaler_cls = self.SCALERS[self.scale]
            self.scaler = scaler_cls()
            X = self.scaler.fit_transform(X)
        
        # Dimensionality reduction (placeholder for PCA)
        if self.reduce_dim == 'pca':
            raise NotImplementedError("PCA reduction coming soon")
        
        self._fitted = True
        return X
    
    def transform(self, X):
        """Transform new data using fitted preprocessor."""
        if not self._fitted:
            raise RuntimeError("Preprocessor not fitted. Call fit_transform first.")
        
        X = np.asarray(X, dtype=np.float64)
        
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        return X