"""
Benchmark GPU-accelerated KNN vs scikit-learn.
Tests both classification and regression across dataset sizes.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from algorithms.knn import KNN


def benchmark_classification(dataset_sizes, n_features=10, n_classes=3, n_neighbors=5, n_runs=3):
    """
    Benchmark KNN classification: GPU vs CPU.

    Returns:
    --------
    sizes, gpu_times, cpu_times : lists
    """
    gpu_times = []
    cpu_times = []

    print(f"Benchmarking KNN Classification: k={n_neighbors}, {n_features} features, {n_runs} runs each")
    print("=" * 70)

    for n_samples in dataset_sizes:
        print(f"\nDataset size: {n_samples:,} samples")

        # Generate data
        X, y = make_classification(
            n_samples=n_samples, n_features=n_features,
            n_classes=n_classes, n_informative=n_features - 2,
            random_state=42
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # GPU benchmark
        gpu_run_times = []
        for _ in range(n_runs):
            knn = KNN(n_neighbors=n_neighbors, task='classification')
            knn.fit(X_train, y_train)
            start = time.perf_counter()
            knn.predict(X_test)
            gpu_run_times.append(time.perf_counter() - start)
        gpu_time = np.mean(gpu_run_times)
        gpu_times.append(gpu_time)

        # CPU benchmark (scikit-learn)
        cpu_run_times = []
        for _ in range(n_runs):
            sk_knn = KNeighborsClassifier(n_neighbors=n_neighbors)
            sk_knn.fit(X_train, y_train)
            start = time.perf_counter()
            sk_knn.predict(X_test)
            cpu_run_times.append(time.perf_counter() - start)
        cpu_time = np.mean(cpu_run_times)
        cpu_times.append(cpu_time)

        speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')
        print(f"  GPU: {gpu_time:.4f}s | CPU: {cpu_time:.4f}s | Speedup: {speedup:.2f}x")

    return dataset_sizes, gpu_times, cpu_times


def benchmark_regression(dataset_sizes, n_features=10, n_neighbors=5, n_runs=3):
    """
    Benchmark KNN regression: GPU vs CPU.
    """
    gpu_times = []
    cpu_times = []

    print(f"\nBenchmarking KNN Regression: k={n_neighbors}, {n_features} features, {n_runs} runs each")
    print("=" * 70)

    for n_samples in dataset_sizes:
        print(f"\nDataset size: {n_samples:,} samples")

        X, y = make_regression(
            n_samples=n_samples, n_features=n_features,
            noise=10, random_state=42
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # GPU
        gpu_run_times = []
        for _ in range(n_runs):
            knn = KNN(n_neighbors=n_neighbors, task='regression')
            knn.fit(X_train, y_train)
            start = time.perf_counter()
            knn.predict(X_test)
            gpu_run_times.append(time.perf_counter() - start)
        gpu_time = np.mean(gpu_run_times)
        gpu_times.append(gpu_time)

        # CPU
        cpu_run_times = []
        for _ in range(n_runs):
            sk_knn = KNeighborsRegressor(n_neighbors=n_neighbors)
            sk_knn.fit(X_train, y_train)
            start = time.perf_counter()
            sk_knn.predict(X_test)
            cpu_run_times.append(time.perf_counter() - start)
        cpu_time = np.mean(cpu_run_times)
        cpu_times.append(cpu_time)

        speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')
        print(f"  GPU: {gpu_time:.4f}s | CPU: {cpu_time:.4f}s | Speedup: {speedup:.2f}x")

    return dataset_sizes, gpu_times, cpu_times


def plot_results(sizes, gpu_clf, cpu_clf, gpu_reg, cpu_reg,
                 save_path="benchmarks/knn_benchmark.png"):
    """Plot classification and regression benchmarks side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Classification
    ax = axes[0]
    ax.plot(sizes, gpu_clf, 'o-', label='GPU (PyTorch)', linewidth=2, markersize=8, color='#1f77b4')
    ax.plot(sizes, cpu_clf, 's-', label='CPU (scikit-learn)', linewidth=2, markersize=8, color='#ff7f0e')
    ax.set_xlabel('Dataset Size (samples)', fontsize=11)
    ax.set_ylabel('Time (seconds)', fontsize=11)
    ax.set_title('KNN Classification: GPU vs CPU', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')

    # Regression
    ax = axes[1]
    ax.plot(sizes, gpu_reg, 'o-', label='GPU (PyTorch)', linewidth=2, markersize=8, color='#1f77b4')
    ax.plot(sizes, cpu_reg, 's-', label='CPU (scikit-learn)', linewidth=2, markersize=8, color='#ff7f0e')
    ax.set_xlabel('Dataset Size (samples)', fontsize=11)
    ax.set_ylabel('Time (seconds)', fontsize=11)
    ax.set_title('KNN Regression: GPU vs CPU', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\nPlot saved to: {save_path}")
    plt.show()


def print_table(sizes, gpu_clf, cpu_clf, gpu_reg, cpu_reg):
    """Print markdown-style results table."""
    print("\n" + "=" * 80)
    print("KNN BENCHMARK RESULTS")
    print("=" * 80)
    print(f"{'Size':>10} | {'Clf GPU':>10} | {'Clf CPU':>10} | {'Clf Speedup':>12} | {'Reg GPU':>10} | {'Reg CPU':>10} | {'Reg Speedup':>12}")
    print("-" * 80)
    for i, n in enumerate(sizes):
        clf_spd = cpu_clf[i] / gpu_clf[i] if gpu_clf[i] > 0 else 0
        reg_spd = cpu_reg[i] / gpu_reg[i] if gpu_reg[i] > 0 else 0
        print(f"{n:>10,} | {gpu_clf[i]:>10.4f} | {cpu_clf[i]:>10.4f} | {clf_spd:>12.2f}x | {gpu_reg[i]:>10.4f} | {cpu_reg[i]:>10.4f} | {reg_spd:>12.2f}x")
    print("=" * 80)


if __name__ == "__main__":
    sizes = [1000, 5000, 10000, 50000, 100000]

    # Classification benchmark
    _, gpu_clf, cpu_clf = benchmark_classification(
        sizes, n_features=10, n_classes=3, n_neighbors=5, n_runs=3
    )

    # Regression benchmark
    _, gpu_reg, cpu_reg = benchmark_regression(
        sizes, n_features=10, n_neighbors=5, n_runs=3
    )

    # Print table
    print_table(sizes, gpu_clf, cpu_clf, gpu_reg, cpu_reg)

    # Plot
    plot_results(sizes, gpu_clf, cpu_clf, gpu_reg, cpu_reg)