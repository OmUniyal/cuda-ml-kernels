# CUDA ML Kernels

GPU-accelerated machine learning benchmarking framework. Built with PyTorch for maximum compatibility and performance.

> **Status:** Early development. K-Means clustering, KNN and PCA are implemented. More algorithms and features coming soon.

---

## Features

### Core Framework
- [x] **Modular architecture** — algorithms separated from GPU kernels
- [x] **Universal data loader** — CSV, NumPy arrays, images, text
- [x] **Auto-preprocessing pipeline** — scaling, encoding, dimensionality reduction
- [ ] **Plugin system** — users can register custom algorithms

### Algorithms
- [x] **K-Means Clustering** — GPU-accelerated, 2-3.5x speedup over sklearn
- [x] **K-Nearest Neighbors (KNN)** — classification and regression with GPU/CPU auto-dispatch
- [x] **Principal Component Analysis (PCA)** — randomized SVD on GPU, 1.5–6x speedup
- [ ] **Linear Regression** — normal equation and gradient descent

### GPU & Performance
- [x] **GPU Acceleration** — PyTorch CUDA backend
- [x] **CPU Fallback** — automatic when no GPU available
- [x] **Benchmark suite** — GPU vs CPU timing comparisons for K-Means, KNN, and PCA
- [ ] **Multi-GPU support** — data parallelism across GPUs
- [ ] **Memory optimization** — chunked processing for large datasets

### Evaluation & Visualization
- [x] **Clustering metrics** — silhouette score, ARI, NMI, inertia
- [x] **Classification metrics** — accuracy, precision, recall, F1, confusion matrix
- [x] **Regression metrics** — MSE, RMSE, MAE, R²
- [x] **Elbow plot** — optimal k finder
- [x] **Cluster scatter plots** — 2D/3D visualization
- [x] **Prediction visualization** — PCA projection with correct/wrong highlights
- [ ] **Benchmark charts** — performance comparison graphs

### User Experience
- [x] **scikit-learn compatible API** — familiar `fit()`, `predict()` interface
- [x] **Unit tests** — validated against scikit-learn for K-Means and KNN
- [x] **Web UI** — Streamlit interface with K-Means, KNN, and PCA support
- [ ] **PyPI package** — `pip install cuda-ml-kernels`
- [ ] **Jupyter notebook examples** — interactive tutorials

---

## Known Limitations

- **GPU Required for Speedup**: CUDA-capable NVIDIA GPU needed for acceleration. CPU fallback works but is slower than scikit-learn.
- **Limited Algorithms**: Currently K-Means, KNN, and PCA. More algorithms in development.
- **Windows Native Issues**: PyTorch CUDA may trigger antivirus (McAfee) on Windows. WSL2 recommended.
- **Memory Bound**: Large datasets (>1M samples) may exceed consumer GPU VRAM (8GB).
- **Manual Preprocessing**: Categorical data must be encoded before feeding. Auto-preprocessing planned.

---

## Tech Stack

- **PyTorch** — GPU-accelerated tensor operations
- **scikit-learn** — CPU baseline implementations
- **matplotlib & seaborn** — Visualization and benchmarking
- **pandas** — Data handling
- **pytest** — Unit testing
- **Streamlit** — Web UI
- **GitHub Actions** — CI/CD

---

## Setup

```bash
# Clone the repository
git clone https://github.com/OmUniyal/cuda-ml-kernels.git
cd cuda-ml-kernels

# Create virtual environment
python -m venv .venv

# Activate (Linux/WSL)
source .venv/bin/activate

# Activate (Windows)
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in editable mode
pip install -e .
```

---

## Quick Start

### K-Means Clustering

```python
from algorithms.kmeans import KMeans
import numpy as np

# Generate sample data
np.random.seed(42)
X = np.random.randn(10000, 10)

# Fit GPU-accelerated K-Means
model = KMeans(n_clusters=5, random_state=42, verbose=True)
model.fit(X)

print(f"Labels: {model.labels}")
print(f"Inertia: {model.inertia:.2f}")
print(f"Iterations: {model.n_iter}")
```

### K-Nearest Neighbors

```python
from algorithms.knn import KNN
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate sample data
X, y = make_classification(n_samples=5000, n_features=10, n_classes=3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Classification
knn = KNN(n_neighbors=5, task='classification')
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)
accuracy = knn.score(X_test, y_test)
print(f"Accuracy: {accuracy:.4f}")

# Regression
from sklearn.datasets import make_regression
X, y = make_regression(n_samples=5000, n_features=10, noise=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn = KNN(n_neighbors=5, task='regression')
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)
r2 = knn.score(X_test, y_test)
print(f"R² Score: {r2:.4f}")
```


### Principal Component Analysis

```python
from algorithms.pca import PCA
import numpy as np

# Generate high-dimensional data
np.random.seed(42)
X = np.random.randn(10000, 100)

# Fit GPU-accelerated PCA
pca = PCA(n_components=10, random_state=42)
X_reduced = pca.fit_transform(X)

print(f"Original shape: {X.shape}")
print(f"Reduced shape: {X_reduced.shape}")
print(f"Explained variance ratio: {pca.explained_variance_ratio_[:3]}")
```

---

## Benchmarks

Run performance comparisons:

```bash
# K-Means benchmark
python benchmarks/benchmark_kmeans.py

# KNN benchmark
python benchmarks/benchmark_knn.py

# PCA benchmark
python benchmarks/benchmark_pca.py
```

### K-Means Results

Sample results on RTX 5060 Laptop GPU:

| Dataset Size | GPU Time | CPU Time | Speedup |
|-------------|----------|----------|---------|
| 1,000 | 0.28s | 0.12s | 0.42x |
| 5,000 | 0.17s | 0.36s | 2.10x |
| 10,000 | 0.21s | 0.51s | 2.44x |
| 50,000 | 0.23s | 0.80s | 3.50x |
| 100,000 | 0.32s | 0.74s | 2.34x |

*GPU speedup becomes significant at 5,000+ samples.*

### KNN Results

Sample results on RTX 5060 Laptop GPU:

**Classification:**

| Dataset Size | GPU Time | CPU Time | Speedup |
| ------------ | -------- | -------- | ------- |
| 1,000        | 0.0115s  | 0.0037s  | 0.32x   |
| 5,000        | 0.0389s  | 0.0271s  | 0.70x   |
| 10,000       | 0.0905s  | 0.0827s  | 0.91x   |
| 50,000       | 0.7056s  | 0.5452s  | 0.77x   |
| 100,000      | 1.7684s  | 1.6063s  | 0.91x   |

**Regression:**

| Dataset Size | GPU Time | CPU Time | Speedup |
| ------------ | -------- | -------- | ------- |
| 1,000        | 0.0042s  | 0.0058s  | 1.37x   |
| 5,000        | 0.0784s  | 0.0532s  | 0.68x   |
| 10,000       | 0.2218s  | 0.2204s  | 0.99x   |
| 50,000       | 2.3266s  | 2.4417s  | 1.05x   |
| 100,000      | 7.7307s  | 8.3121s  | 1.08x   |

*KNN is memory-bandwidth bound rather than compute bound. GPU acceleration is modest on consumer laptop GPUs due to PCIe transfer overhead. Regression sees slight benefits at 50K+ samples; classification performs comparably to CPU.*


### PCA Results

Run the benchmark on your machine:

```bash
python benchmarks/benchmark_pca.py
```

Results on RTX 5060 Laptop GPU (10 components, 100 features):

| Dataset Size | GPU Time | CPU Time | Speedup |
|-------------|----------|----------|---------|
| 1,000 | 0.0076s | 0.0060s | 0.80x |
| 5,000 | 0.0085s | 0.0124s | 1.46x |
| 10,000 | 0.0134s | 0.0858s | 6.42x |
| 50,000 | 0.0676s | 0.1852s | 2.74x |
| 100,000 | 0.1496s | 0.3623s | 2.42x |

*PCA shows strong GPU acceleration for medium-to-large datasets due to compute-intensive matrix operations. Speedup becomes significant at 5,000+ samples.*

---

## Project Structure

```
cuda-ml-kernels/
├── src/
│   ├── algorithms/         # ML algorithm implementations
│   ├── cuda_kernels/       # GPU operations
│   ├── core/               # Framework backbone (data loader, preprocessor, evaluator)
│   ├── ui/                 # Streamlit web interface
│   └── utils/              # Helper functions
├── tests/                  # Unit tests (pytest)
├── benchmarks/             # Performance comparisons
├── examples/               # Usage examples (planned)
├── data/                   # User datasets
├── notebooks/              # Exploration notebooks
└── docs/                   # Documentation
```

---

## Contributing

This project is in active development. Check the Features section above to see where help is needed.

---

## License

MIT
