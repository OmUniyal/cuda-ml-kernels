"""
Streamlit UI for CUDA ML Kernels.
Supports: K-Means Clustering & K-Nearest Neighbors (Classification/Regression)
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs, make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)
import seaborn as sns

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.data_loader import DataLoader
from core.preprocessor import Preprocessor
from core.evaluator import ClusteringEvaluator
from algorithms.kmeans import KMeans
from algorithms.knn import KNN

st.set_page_config(
    page_title="CUDA ML Kernels",
    page_icon="🚀",
    layout="wide"
)

st.title("🚀 CUDA ML Kernels")
st.markdown("GPU-accelerated machine learning with PyTorch CUDA")

# ── Initialize Session State ──
def init_state():
    defaults = {
        'data_loaded': False,
        'X': None,
        'y': None,
        'df_preview': None,
        'X_train': None,
        'X_test': None,
        'y_train': None,
        'y_test': None,
        'knn_model': None,
        'knn_predictions': None,
        'knn_metrics': None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ── Sidebar: Data Source ──
with st.sidebar:
    st.header("📁 Data Source")
    data_source = st.radio(
        "Choose data source",
        ["Upload CSV", "Upload NPY", "Generate Synthetic"],
        key="data_source"
    )

    # ── CSV Upload ──
    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader("Choose CSV file", type=['csv'], key="csv_uploader")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.session_state.df_preview = df

            st.subheader("Column Selection")
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            all_cols = df.columns.tolist()

            if len(numeric_cols) == 0:
                st.error("No numeric columns found!")
                st.stop()

            feature_cols = st.multiselect(
                "Feature columns",
                numeric_cols,
                default=numeric_cols,
                key="csv_features"
            )

            target_col = st.selectbox(
                "Target column (optional — needed for KNN)",
                ["None"] + all_cols,
                key="csv_target"
            )

            if st.button("Load Data", key="csv_load"):
                if len(feature_cols) == 0:
                    st.error("Select at least one feature column!")
                    st.stop()

                X = df[feature_cols].values.astype(np.float64)
                y = None
                if target_col != "None":
                    y = df[target_col].values

                st.session_state.X = X
                st.session_state.y = y
                st.session_state.data_loaded = True
                st.success(f"Loaded: {X.shape[0]} samples, {X.shape[1]} features")

    # ── NPY Upload ──
    elif data_source == "Upload NPY":
        uploaded_file = st.file_uploader("Choose NPY/NPZ file", type=['npy', 'npz'], key="npy_uploader")
        if uploaded_file is not None:
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
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

    # ── Synthetic Data ──
    elif data_source == "Generate Synthetic":
        st.subheader("Synthetic Generator")
        synth_type = st.selectbox(
            "Dataset type",
            ["Blobs (Clustering)", "Classification", "Regression"],
            key="synth_type"
        )
        n_samples = st.slider("Samples", 100, 50000, 5000, key="synth_samples")
        n_features = st.slider("Features", 2, 50, 10, key="synth_features")

        if st.button("Generate Data", key="synth_generate"):
            if synth_type == "Blobs (Clustering)":
                X, y = make_blobs(
                    n_samples=n_samples,
                    n_features=n_features,
                    centers=5,
                    random_state=42
                )
            elif synth_type == "Classification":
                X, y = make_classification(
                    n_samples=n_samples,
                    n_features=n_features,
                    n_informative=n_features - 2,
                    n_redundant=2,
                    n_classes=3,
                    random_state=42
                )
            else:  # Regression
                X, y = make_regression(
                    n_samples=n_samples,
                    n_features=n_features,
                    noise=10.0,
                    random_state=42
                )

            st.session_state.X = X.astype(np.float64)
            st.session_state.y = y
            st.session_state.data_loaded = True
            st.success(f"Generated: {X.shape[0]} samples, {X.shape[1]} features")

    # ── Algorithm Selection (only when data loaded) ──
    if st.session_state.data_loaded:
        st.header("⚙️ Algorithm")
        algorithm = st.selectbox(
            "Select algorithm",
            ["K-Means Clustering", "K-Nearest Neighbors"],
            key="algorithm"
        )

        # ── K-Means Settings ──
        if algorithm == "K-Means Clustering":
            n_clusters = st.slider("Number of Clusters (k)", 2, 50, 5, key="kmeans_k")
            scale = st.selectbox("Scaling", ["standard", "minmax", "none"], key="kmeans_scale")
            show_elbow = st.checkbox("Show Elbow Plot", value=True, key="kmeans_elbow")
            run_button = st.button("🚀 Run K-Means", type="primary", key="kmeans_run")

        # ── KNN Settings ──
        elif algorithm == "K-Nearest Neighbors":
            knn_task = st.selectbox(
                "Task",
                ["classification", "regression"],
                key="knn_task"
            )
            n_neighbors = st.slider("K (neighbors)", 1, 50, 5, key="knn_k")
            test_size = st.slider("Test split (%)", 10, 50, 20, key="knn_test") / 100.0
            scale = st.selectbox("Scaling", ["standard", "minmax", "none"], key="knn_scale")
            random_state = st.number_input("Random seed", 0, 9999, 42, key="knn_seed")
            run_button = st.button("🚀 Run KNN", type="primary", key="knn_run")

# ═══════════════════════════════════════════════════════════════
# MAIN AREA
# ═══════════════════════════════════════════════════════════════

# ── Data Preview ──
if st.session_state.data_loaded and st.session_state.df_preview is not None:
    with st.expander("📊 Data Preview"):
        st.dataframe(st.session_state.df_preview.head(20), use_container_width=True)
        st.write(f"Shape: {st.session_state.df_preview.shape}")

# ── K-MEANS CLUSTERING ──
if st.session_state.data_loaded and algorithm == "K-Means Clustering" and 'run_button' in locals() and run_button:
    X = st.session_state.X
    y_true = st.session_state.y

    with st.spinner("Preprocessing..."):
        prep = Preprocessor(scale=scale)
        X_processed = prep.fit_transform(X)

    with st.spinner("Running GPU K-Means..."):
        model = KMeans(n_clusters=n_clusters, random_state=42, verbose=False)
        model.fit(X_processed)

    # Results
    st.header("📈 K-Means Results")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Cluster Visualization (PCA)")
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
            alpha=0.6,
            s=20
        )
        # Plot centroids
        if X_processed.shape[1] > 2:
            centroids_2d = pca.transform(model.centroids)
        else:
            centroids_2d = model.centroids
        ax.scatter(centroids_2d[:, 0], centroids_2d[:, 1],
                   c='black', marker='X', s=200, edgecolors='white', linewidths=2,
                   label='Centroids')
        ax.set_xlabel("PC1" if X_processed.shape[1] > 2 else "Feature 1")
        ax.set_ylabel("PC2" if X_processed.shape[1] > 2 else "Feature 2")
        ax.set_title("Clusters (2D projection via PCA)")
        ax.legend()
        plt.colorbar(scatter, ax=ax, label='Cluster')
        st.pyplot(fig)

    with col2:
        st.subheader("Metrics")
        metrics = ClusteringEvaluator.evaluate(
            X_processed, model.labels,
            centroids=model.centroids,
            true_labels=y_true
        )
        metrics_df = pd.DataFrame([
            {"Metric": k, "Value": f"{v:.4f}"}
            for k, v in metrics.items()
        ])
        st.table(metrics_df)

        st.metric("Iterations to converge", model.n_iter)
        st.metric("Inertia (SSE)", f"{model.inertia:.2f}")
        st.metric("Samples", X.shape[0])
        st.metric("Features", X.shape[1])

    # Elbow plot
    if show_elbow:
        st.subheader("Elbow Plot (Find Optimal k)")
        max_k = min(16, len(X) // 10)
        if max_k < 3:
            st.info("Dataset too small for elbow plot.")
        else:
            k_range = range(2, max_k)
            inertias = []
            progress_bar = st.progress(0, text="Computing elbow plot...")
            for i, k in enumerate(k_range):
                kmeans = KMeans(n_clusters=k, random_state=42, verbose=False)
                kmeans.fit(X_processed)
                inertias.append(kmeans.inertia)
                progress_bar.progress((i + 1) / len(k_range), text=f"k={k} done")
            progress_bar.empty()

            fig2, ax2 = plt.subplots(figsize=(10, 5))
            ax2.plot(k_range, inertias, 'bo-', markersize=8, linewidth=2)
            ax2.axvline(x=n_clusters, color='r', linestyle='--', alpha=0.5, label=f'Current k={n_clusters}')
            ax2.set_xlabel("Number of Clusters (k)")
            ax2.set_ylabel("Inertia (SSE)")
            ax2.set_title("Elbow Method for Optimal k")
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            st.pyplot(fig2)

    st.success("K-Means complete! 🎉")

# ── K-NEAREST NEIGHBORS ──
if st.session_state.data_loaded and algorithm == "K-Nearest Neighbors" and 'run_button' in locals() and run_button:
    X = st.session_state.X
    y = st.session_state.y

    # Validation
    if y is None:
        st.error("❌ KNN requires a target column. Please reload your data with a target selected, or use a synthetic dataset with labels.")
        st.stop()

    # Ensure labels are integers for classification
    if knn_task == "classification":
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y = le.fit_transform(y.astype(str))
        st.info(f"Encoded {len(le.classes_)} classes: {list(le.classes_)}")

    with st.spinner("Preprocessing & Splitting..."):
        prep = Preprocessor(scale=scale)
        X_processed = prep.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=test_size, random_state=int(random_state),
            stratify=y if knn_task == "classification" else None
        )
        st.session_state.X_train = X_train
        st.session_state.X_test = X_test
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test

    with st.spinner(f"Running GPU KNN ({knn_task})..."):
        model = KNN(n_neighbors=n_neighbors, task=knn_task)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        st.session_state.knn_model = model
        st.session_state.knn_predictions = predictions

    st.header(f"📈 KNN {knn_task.title()} Results")

    # ── Classification Results ──
    if knn_task == "classification":
        acc = accuracy_score(y_test, predictions)
        prec = precision_score(y_test, predictions, average='weighted', zero_division=0)
        rec = recall_score(y_test, predictions, average='weighted', zero_division=0)
        f1 = f1_score(y_test, predictions, average='weighted', zero_division=0)

        st.session_state.knn_metrics = {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1
        }

        # Metrics row
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{acc:.4f}")
        m2.metric("Precision", f"{prec:.4f}")
        m3.metric("Recall", f"{rec:.4f}")
        m4.metric("F1 Score", f"{f1:.4f}")

        # Confusion Matrix
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, predictions)
        fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                    xticklabels=le.classes_, yticklabels=le.classes_)
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("Actual")
        ax_cm.set_title("Confusion Matrix")
        st.pyplot(fig_cm)

        # Classification Report
        st.subheader("Detailed Report")
        report = classification_report(y_test, predictions, target_names=[str(c) for c in le.classes_], output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.format("{:.4f}"), use_container_width=True)

        # Visualization (PCA for 2D scatter)
        st.subheader("Prediction Visualization (PCA)")
        if X_processed.shape[1] > 2:
            pca = PCA(n_components=2)
            X_test_2d = pca.fit_transform(X_test)
        else:
            X_test_2d = X_test

        fig_viz, ax_viz = plt.subplots(figsize=(10, 7))
        correct = (predictions == y_test)
        colors = np.where(correct, 'green', 'red')
        scatter = ax_viz.scatter(X_test_2d[:, 0], X_test_2d[:, 1],
                                  c=colors, alpha=0.6, s=30, edgecolors='black', linewidth=0.3)
        ax_viz.set_title("Test Set Predictions — Green=Correct, Red=Wrong")
        ax_viz.set_xlabel("PC1" if X_processed.shape[1] > 2 else "Feature 1")
        ax_viz.set_ylabel("PC2" if X_processed.shape[1] > 2 else "Feature 2")

        # Add class centroids as text
        for cls in np.unique(y_test):
            mask = y_test == cls
            if mask.sum() > 0:
                cx = X_test_2d[mask, 0].mean()
                cy = X_test_2d[mask, 1].mean()
                ax_viz.annotate(str(le.classes_[cls]), (cx, cy), fontsize=12, fontweight='bold',
                                ha='center', va='center',
                                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        st.pyplot(fig_viz)

    # ── Regression Results ──
    else:
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        st.session_state.knn_metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2_score': r2
        }

        # Metrics row
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("MSE", f"{mse:.4f}")
        m2.metric("RMSE", f"{rmse:.4f}")
        m3.metric("MAE", f"{mae:.4f}")
        m4.metric("R² Score", f"{r2:.4f}")

        # Actual vs Predicted scatter
        st.subheader("Actual vs Predicted")
        fig_reg, ax_reg = plt.subplots(figsize=(8, 8))
        ax_reg.scatter(y_test, predictions, alpha=0.5, s=30, edgecolors='black', linewidth=0.3)
        min_val = min(y_test.min(), predictions.min())
        max_val = max(y_test.max(), predictions.max())
        ax_reg.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        ax_reg.set_xlabel("Actual")
        ax_reg.set_ylabel("Predicted")
        ax_reg.set_title(f"Actual vs Predicted (R² = {r2:.4f})")
        ax_reg.legend()
        ax_reg.grid(True, alpha=0.3)
        st.pyplot(fig_reg)

        # Residuals plot
        st.subheader("Residuals Distribution")
        residuals = y_test - predictions
        fig_res, ax_res = plt.subplots(figsize=(10, 5))
        ax_res.hist(residuals, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
        ax_res.axvline(x=0, color='red', linestyle='--', lw=2, label='Zero Error')
        ax_res.set_xlabel("Residual (Actual - Predicted)")
        ax_res.set_ylabel("Frequency")
        ax_res.set_title("Residuals Distribution")
        ax_res.legend()
        st.pyplot(fig_res)

    # ── Common KNN Info ──
    st.subheader("Training Info")
    info_col1, info_col2, info_col3 = st.columns(3)
    info_col1.metric("Train samples", len(X_train))
    info_col2.metric("Test samples", len(X_test))
    info_col3.metric("Features", X.shape[1])

    st.success(f"KNN {knn_task} complete! 🎉")

# ── No data loaded yet ──
if not st.session_state.data_loaded:
    st.info("👈 Use the sidebar to load or generate data to get started.")
    st.markdown("""
    ### Supported Algorithms
    - **K-Means Clustering** — Unsupervised clustering with GPU acceleration
    - **K-Nearest Neighbors** — Classification or regression with GPU-accelerated distance computation

    ### Quick Tips
    - For **K-Means**: Upload any numeric dataset (no target needed)
    - For **KNN**: Make sure to select a target column, or use synthetic Classification/Regression data
    - GPU speedup is most noticeable with 5,000+ samples
    """)

# ── Footer ──
st.markdown("---")
st.markdown(
    "Built with ❤️ using [CUDA ML Kernels](https://github.com/OmUniyal/cuda-ml-kernels)"
)