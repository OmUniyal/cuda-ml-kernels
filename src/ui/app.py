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


# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
    st.session_state.X = None
    st.session_state.y = None
    st.session_state.df_preview = None


# Sidebar
with st.sidebar:
    st.header("Settings")
    
    # Data source
    data_source = st.radio(
        "Data Source",
        ["Upload CSV", "Upload NPY", "Generate Synthetic"]
    )
    
    # CSV Upload
    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.session_state.df_preview = df
            
            st.subheader("Column Selection")
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) == 0:
                st.error("No numeric columns found!")
                st.stop()
            
            feature_cols = st.multiselect(
                "Feature columns",
                numeric_cols,
                default=numeric_cols
            )
            
            target_col = st.selectbox(
                "Target column (optional)",
                ["None"] + df.columns.tolist()
            )
            
            if st.button("Load Data"):
                if len(feature_cols) == 0:
                    st.error("Select at least one feature column!")
                    st.stop()
                
                X = df[feature_cols].values
                y = None
                if target_col != "None":
                    y = df[target_col].values
                
                st.session_state.X = X
                st.session_state.y = y
                st.session_state.data_loaded = True
                st.success(f"Loaded: {X.shape[0]} samples, {X.shape[1]} features")
    
    # NPY Upload
    elif data_source == "Upload NPY":
        uploaded_file = st.file_uploader("Choose NPY/NPZ file", type=['npy', 'npz'])
        if uploaded_file is not None:
            # Save temporarily
            temp_path = "/tmp/uploaded_data.npy"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            
            try:
                X, y = DataLoader.load(temp_path)
                st.session_state.X = X
                st.session_state.y = y
                st.session_state.data_loaded = True
                st.success(f"Loaded: {X.shape[0]} samples, {X.shape[1]} features")
            except Exception as e:
                st.error(f"Error loading file: {e}")
            
            # Cleanup
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    # Synthetic Data
    elif data_source == "Generate Synthetic":
        n_samples = st.slider("Samples", 100, 20000, 5000)
        n_features = st.slider("Features", 2, 50, 10)
        
        if st.button("Generate Data"):
            X, y = make_blobs(
                n_samples=n_samples,
                n_features=n_features,
                centers=5,
                random_state=42
            )
            st.session_state.X = X
            st.session_state.y = y
            st.session_state.data_loaded = True
            st.success(f"Generated: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Only show algorithm settings if data is loaded
    if st.session_state.data_loaded:
        st.header("Algorithm")
        algorithm = st.selectbox("Algorithm", ["K-Means"])
        n_clusters = st.slider("Number of Clusters", 2, 20, 5)
        scale = st.selectbox("Scaling", ["standard", "minmax", "none"])
        run_button = st.button("🚀 Run Clustering", type="primary")


# Main area
if st.session_state.data_loaded and st.session_state.df_preview is not None:
    with st.expander("Data Preview"):
        st.dataframe(st.session_state.df_preview.head(20))
        st.write(f"Shape: {st.session_state.df_preview.shape}")

if st.session_state.data_loaded and 'run_button' in locals() and run_button:
    X = st.session_state.X
    y_true = st.session_state.y
    
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
        
        n_components_viz = min(2, X_processed.shape[1])
        if X_processed.shape[1] > 2:
            pca = PCA(n_components=2)
            X_2d = pca.fit_transform(X_processed)
        else:
            X_2d = X_processed
        
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(
            X_2d[:, 0],
            X_2d[:, 1] if X_2d.shape[1] > 1 else np.zeros(len(X_2d)),
            c=model.labels,
            cmap='tab10',
            alpha=0.6
        )
        ax.set_xlabel("PC1" if X_processed.shape[1] > 2 else "Feature 1")
        ax.set_ylabel("PC2" if X_processed.shape[1] > 2 else "Feature 2")
        ax.set_title("Clusters (2D projection)")
        plt.colorbar(scatter, ax=ax)
        st.pyplot(fig)
    
    with col2:
        st.subheader("Metrics")
        
        metrics = ClusteringEvaluator.evaluate(
            X_processed,
            model.labels,
            centroids=model.centroids,
            true_labels=y_true
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
    
    max_k = min(16, len(X) // 10)
    k_range = range(2, max_k)
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