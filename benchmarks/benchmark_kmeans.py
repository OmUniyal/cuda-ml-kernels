"""
Benchmark GPU-accelerated K-Means vs scikit-learn.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans as SklearnKMeans
from algorithms.kmeans import KMeans


def benchmark(dataset_sizes, n_clusters=5, n_features=10, n_runs=3):
    """
    Benchmark GPU K-Means vs scikit-learn across different dataset sizes.
    
    Parameters:
    -----------
    dataset_sizes : list of ints
        Number of samples to test
    n_clusters : int
    n_features : int
    n_runs : int
        Number of runs per size (for averaging)
    """
    gpu_times = []
    cpu_times = []
    
    print(f"Benchmarking K-Means: {n_clusters} clusters, {n_features} features, {n_runs} runs each")
    print("=" * 60)
    
    for n_samples in dataset_sizes:
        print(f"\nDataset size: {n_samples:,} samples")
        
        # Generate random data
        np.random.seed(42)
        X = np.random.randn(n_samples, n_features).astype(np.float64)
        
        # Benchmark GPU
        gpu_run_times = []
        for _ in range(n_runs):
            gpu_kmeans = KMeans(n_clusters=n_clusters, random_state=42, max_iter=100)
            start = time.perf_counter()
            gpu_kmeans.fit(X)
            gpu_run_times.append(time.perf_counter() - start)
        gpu_time = np.mean(gpu_run_times)
        gpu_times.append(gpu_time)
        
        # Benchmark CPU (scikit-learn)
        cpu_run_times = []
        for _ in range(n_runs):
            cpu_kmeans = SklearnKMeans(n_clusters=n_clusters, random_state=42, max_iter=100, n_init=1)
            start = time.perf_counter()
            cpu_kmeans.fit(X)
            cpu_run_times.append(time.perf_counter() - start)
        cpu_time = np.mean(cpu_run_times)
        cpu_times.append(cpu_time)
        
        speedup = cpu_time / gpu_time
        print(f"  GPU: {gpu_time:.4f}s | CPU: {cpu_time:.4f}s | Speedup: {speedup:.2f}x")
    
    return dataset_sizes, gpu_times, cpu_times


def plot_results(dataset_sizes, gpu_times, cpu_times, save_path="benchmarks/kmeans_benchmark.png"):
    """Plot benchmark results."""
    plt.figure(figsize=(10, 6))
    
    plt.plot(dataset_sizes, gpu_times, 'o-', label='GPU (PyTorch)', linewidth=2, markersize=8)
    plt.plot(dataset_sizes, cpu_times, 's-', label='CPU (scikit-learn)', linewidth=2, markersize=8)
    
    plt.xlabel('Dataset Size (samples)', fontsize=12)
    plt.ylabel('Time (seconds)', fontsize=12)
    plt.title('K-Means Clustering: GPU vs CPU Performance', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\nPlot saved to: {save_path}")
    plt.show()


if __name__ == "__main__":
    # Benchmark on increasing dataset sizes
    sizes = [1000, 5000, 10000, 50000, 100000]
    
    sizes, gpu_t, cpu_t = benchmark(sizes, n_clusters=5, n_features=10, n_runs=3)
    plot_results(sizes, gpu_t, cpu_t)