"""
Streamlit UI for CUDA ML Kernels.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.data_loader import DataLoader
from core.preprocessor import Preprocessor
from core.evaluator import ClusteringEvaluator
from algorithms.kmeans import KMeans


st.set_page_config(
    page_title="CUDA ML Kernels",
    page_icon="🚀",
    layout="wide"
)

st.title("🚀 CUDA ML Kernels")
st.markdown("GPU-accelerated clustering with PyTorch")


# Sidebar
with st.sidebar:
    st.header("Settings")
    
    # Data source
    data_source = st.radio(
        "Data Source",
        ["Upload CSV", "Upload NPY", "Generate Synthetic"]
    )
    
    # Algorithm
    algorithm = st.selectbox(
        "Algorithm",
        ["K-Means"]
    )
    
    # Parameters
    n_clusters = st.slider("Number of Clusters", 2, 20, 5)
    scale = st.selectbox("Scaling", ["standard", "minmax", "none"])
    
    run_button = st.button("🚀 Run Clustering", type="primary")


# Main area
if run_button:
    with st.spinner("Loading data..."):
        if data_source == "Generate Synthetic":
            X, true_labels = make_blobs(
                n_samples=5000,
                n_features=10,
                centers=n_clusters,
                random_state=42
            )
            st.success(f"Generated synthetic data: {X.shape}")
        else:
            st.info("Upload feature coming soon!")
            st.stop()
    
    with st.spinner("Preprocessing..."):
        prep = Preprocessor(scale=scale)
        X_processed = prep.fit_transform(X)
    
    with st.spinner("Running GPU K-Means..."):
        model = KMeans(n_clusters=n_clusters, random_state=42, verbose=False)
        model.fit(X_processed)
    
    # Results
    st.header("Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Cluster Visualization (PCA)")
        
        # Reduce to 2D for plotting
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X_processed)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(
            X_2d[:, 0],
            X_2d[:, 1],
            c=model.labels,
            cmap='tab10',
            alpha=0.6
        )
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title("Clusters (PCA 2D projection)")
        plt.colorbar(scatter, ax=ax)
        st.pyplot(fig)
    
    with col2:
        st.subheader("Metrics")
        
        metrics = ClusteringEvaluator.evaluate(
            X_processed,
            model.labels,
            centroids=model.centroids,
            true_labels=true_labels if data_source == "Generate Synthetic" else None
        )
        
        metrics_df = pd.DataFrame([
            {"Metric": k, "Value": f"{v:.4f}"}
            for k, v in metrics.items()
        ])
        st.table(metrics_df)
        
        st.metric("Iterations", model.n_iter)
        st.metric("Inertia", f"{model.inertia:.2f}")
    
    # Elbow plot
    st.subheader("Elbow Plot (Find Optimal k)")
    
    k_range = range(2, min(16, len(X) // 10))
    inertias = []
    
    progress_bar = st.progress(0)
    for i, k in enumerate(k_range):
        kmeans = KMeans(n_clusters=k, random_state=42, verbose=False)
        kmeans.fit(X_processed)
        inertias.append(kmeans.inertia)
        progress_bar.progress((i + 1) / len(k_range))
    
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(k_range, inertias, 'bo-')
    ax2.set_xlabel("Number of Clusters (k)")
    ax2.set_ylabel("Inertia")
    ax2.set_title("Elbow Method")
    ax2.grid(True, alpha=0.3)
    st.pyplot(fig2)
    
    st.success("Done! 🎉")


# Footer
st.markdown("---")
st.markdown(
    "Built with ❤️ using [CUDA ML Kernels]"
    "(https://github.com/OmUniyal/cuda-ml-kernels)"
)