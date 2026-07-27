# CUDA ML Kernels

GPU-accelerated implementations of classic machine learning algorithms using CUDA and Python. Built from scratch for learning, benchmarking, and reuse.

## Overview

This project implements fundamental ML algorithms with custom CUDA kernels, benchmarking them against CPU and standard library implementations on real datasets.

## Planned Algorithms

- [ ] K-Means Clustering
- [ ] K-Nearest Neighbors (KNN)
- [ ] Linear Regression
- [ ] Principal Component Analysis (PCA)

## Tech Stack

- **CuPy** — CUDA arrays with NumPy-like API
- **Numba CUDA** — Python-native CUDA kernel writing
- **PyTorch** — Baseline comparisons and dataset loading
- **scikit-learn** — CPU baseline implementations
- **pytest** — Unit testing
- **GitHub Actions** — CI/CD

## Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/cuda-ml-kernels.git
cd cuda-ml-kernels

# Create virtual environment
python -m venv .venv

# Activate (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

```
## Project Structure

cuda-ml-kernels/
├── src/
│   ├── cuda_kernels/      # Raw CUDA kernels
│   ├── algorithms/         # ML algorithm implementations
│   └── utils/              # Helper functions
├── tests/                  # Unit tests
├── benchmarks/             # Performance comparison scripts
├── data/                   # Datasets
├── notebooks/              # Exploration notebooks
└── docs/                   # Documentation
```