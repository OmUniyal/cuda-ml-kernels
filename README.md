# CUDA ML Kernels

GPU-accelerated machine learning benchmarking framework. Built with PyTorch for maximum compatibility and performance.

> **Status:** Early development. K-Means clustering is implemented and benchmarked. More algorithms and features coming soon.

---

## Features

### Core Framework
- [x] **Modular architecture** — algorithms separated from GPU kernels
- [x] **Universal data loader** — CSV, NumPy arrays, images, text
- [x] **Auto-preprocessing pipeline** — scaling, encoding, dimensionality reduction
- [ ] **Plugin system** — users can register custom algorithms

### Algorithms
- [x] **K-Means Clustering** — GPU-accelerated, 2-3.5x speedup over sklearn
- [ ] **K-Nearest Neighbors (KNN)** — brute-force and KD-tree variants
- [ ] **Principal Component Analysis (PCA)** — randomized SVD on GPU
- [ ] **Linear Regression** — normal equation and gradient descent

### GPU & Performance
- [x] **GPU Acceleration** — PyTorch CUDA backend
- [x] **CPU Fallback** — automatic when no GPU available
- [x] **Benchmark suite** — GPU vs CPU timing comparisons
- [ ] **Multi-GPU support** — data parallelism across GPUs
- [ ] **Memory optimization** — chunked processing for large datasets

### Evaluation & Visualization
- [x] **Clustering metrics** — silhouette score, ARI, NMI, inertia
- [x] **Elbow plot** — optimal k finder
- [x] **Cluster scatter plots** — 2D/3D visualization
- [ ] **Benchmark charts** — performance comparison graphs

### User Experience
- [x] **scikit-learn compatible API** — familiar `fit()`, `predict()` interface
- [x] **Unit tests** — validated against scikit-learn
- [x] **Web UI** — Streamlit interface
- [ ] **PyPI package** — `pip install cuda-ml-kernels`
- [ ] **Jupyter notebook examples** — interactive tutorials

---

## Known Limitations

- **GPU Required for Speedup**: CUDA-capable NVIDIA GPU needed for acceleration. CPU fallback works but is slower than scikit-learn.
- **Single Algorithm**: Currently only K-Means. More algorithms in development.
- **Windows Native Issues**: PyTorch CUDA may trigger antivirus (McAfee) on Windows. WSL2 recommended.
- **Memory Bound**: Large datasets (>1M samples) may exceed consumer GPU VRAM (8GB).
- **Manual Preprocessing**: Categorical data must be encoded before feeding. Auto-preprocessing planned.
- **No Web UI Yet**: Command-line only for now. Web interface coming later.

---

## Tech Stack

- **PyTorch** — GPU-accelerated tensor operations
- **scikit-learn** — CPU baseline implementations
- **matplotlib** — Visualization and benchmarking
- **pandas** — Data handling
- **pytest** — Unit testing
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

---

## Benchmarks

Run performance comparison:

```bash
python benchmarks/benchmark_kmeans.py
```

Sample results on RTX 5060 Laptop GPU:

| Dataset Size | GPU Time | CPU Time | Speedup |
|-------------|----------|----------|---------|
| 1,000 | 0.28s | 0.12s | 0.42x |
| 5,000 | 0.17s | 0.36s | 2.10x |
| 10,000 | 0.21s | 0.51s | 2.44x |
| 50,000 | 0.23s | 0.80s | 3.50x |
| 100,000 | 0.32s | 0.74s | 2.34x |

*GPU speedup becomes significant at 5,000+ samples.*

---

## Project Structure

```
cuda-ml-kernels/
├── src/
│   ├── algorithms/         # ML algorithm implementations
│   ├── cuda_kernels/       # GPU operations
│   ├── core/               # Framework backbone (planned)
│   └── utils/              # Helper functions
├── tests/                  # Unit tests
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
