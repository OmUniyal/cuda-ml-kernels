"""
Benchmark GPU-accelerated PCA vs scikit-learn.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA as SklearnPCA
from algorithms.pca import PCA


def benchmark(dataset_sizes, n_features=100, n_components=10, n_runs=3):
    """
    Benchmark GPU PCA vs scikit-learn randomized SVD.
    """
    gpu_times = []
    cpu_times = []

    print(f"Benchmarking PCA: {n_components} components, {n_features} features, {n_runs} runs each")
    print("=" * 70)

    for n_samples in dataset_sizes:
        print(f"\nDataset size: {n_samples:,} samples")

        np.random.seed(42)
        X = np.random.randn(n_samples, n_features).astype(np.float64)

        # GPU benchmark
        gpu_run_times = []
        for _ in range(n_runs):
            pca = PCA(n_components=n_components, random_state=42)
            start = time.perf_counter()
            pca.fit(X)
            gpu_run_times.append(time.perf_counter() - start)
        gpu_time = np.mean(gpu_run_times)
        gpu_times.append(gpu_time)

        # CPU benchmark (sklearn randomized)
        cpu_run_times = []
        for _ in range(n_runs):
            pca = SklearnPCA(n_components=n_components, svd_solver='randomized', random_state=42)
            start = time.perf_counter()
            pca.fit(X)
            cpu_run_times.append(time.perf_counter() - start)
        cpu_time = np.mean(cpu_run_times)
        cpu_times.append(cpu_time)

        speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')
        print(f"  GPU: {gpu_time:.4f}s | CPU: {cpu_time:.4f}s | Speedup: {speedup:.2f}x")

    return dataset_sizes, gpu_times, cpu_times


def plot_results(dataset_sizes, gpu_times, cpu_times, save_path="benchmarks/pca_benchmark.png"):
    """Plot benchmark results."""
    plt.figure(figsize=(10, 6))

    plt.plot(dataset_sizes, gpu_times, 'o-', label='GPU (PyTorch)', linewidth=2, markersize=8)
    plt.plot(dataset_sizes, cpu_times, 's-', label='CPU (sklearn randomized)', linewidth=2, markersize=8)

    plt.xlabel('Dataset Size (samples)', fontsize=12)
    plt.ylabel('Time (seconds)', fontsize=12)
    plt.title('PCA: GPU vs CPU Performance', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    plt.yscale('log')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\nPlot saved to: {save_path}")
    plt.show()


if __name__ == "__main__":
    sizes = [1000, 5000, 10000, 50000, 100000]
    sizes, gpu_t, cpu_t = benchmark(sizes, n_features=100, n_components=10, n_runs=3)
    plot_results(sizes, gpu_t, cpu_t)