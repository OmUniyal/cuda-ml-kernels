"""
Universal data loader for multiple formats.
"""

import numpy as np
import pandas as pd
from pathlib import Path


class DataLoader:
    """
    Load data from various sources and convert to numpy arrays.
    
    Supports: CSV, TSV, NumPy (.npy, .npz), and raw numpy arrays.
    Images and text support planned.
    """
    
    @staticmethod
    def from_csv(path, target_column=None, **kwargs):
        """
        Load data from CSV file.
        
        Parameters:
        -----------
        path : str
            Path to CSV file
        target_column : str, optional
            Column name to separate as labels
        **kwargs : passed to pd.read_csv
        
        Returns:
        --------
        X : ndarray — features
        y : ndarray or None — labels if target_column provided
        """
        df = pd.read_csv(path, **kwargs)
        if target_column:
            y = df[target_column].values
            X = df.drop(columns=[target_column]).select_dtypes(include=[np.number]).values
            return X, y
        return df.select_dtypes(include=[np.number]).values, None
    
    @staticmethod
    def from_numpy(path):
        """
        Load .npy or .npz file.
        
        Returns:
        --------
        X : ndarray
        y : ndarray or None
        """
        path = Path(path)
        if path.suffix == '.npy':
            return np.load(path), None
        elif path.suffix == '.npz':
            data = np.load(path)
            keys = list(data.keys())
            if len(keys) == 1:
                return data[keys[0]], None
            elif len(keys) == 2:
                return data[keys[0]], data[keys[1]]
            else:
                return data[keys[0]], None
        else:
            raise ValueError(f"Unsupported format: {path.suffix}")
    
    @staticmethod
    def from_array(X, y=None):
        """
        Wrap existing numpy arrays.
        
        Returns:
        --------
        X : ndarray
        y : ndarray or None
        """
        return np.asarray(X), np.asarray(y) if y is not None else None
    
    @classmethod
    def load(cls, path_or_data, **kwargs):
        """
        Auto-detect and load data from path or array.
        
        Parameters:
        -----------
        path_or_data : str, Path, or ndarray
        **kwargs : format-specific options
        
        Returns:
        --------
        X : ndarray
        y : ndarray or None
        """
        if isinstance(path_or_data, (str, Path)):
            path = Path(path_or_data)
            if path.suffix == '.csv':
                return cls.from_csv(path, **kwargs)
            elif path.suffix in ('.npy', '.npz'):
                return cls.from_numpy(path)
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}")
        elif isinstance(path_or_data, np.ndarray):
            return cls.from_array(path_or_data, **kwargs)
        else:
            raise TypeError(f"Unsupported data type: {type(path_or_data)}")