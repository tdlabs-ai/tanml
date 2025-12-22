# tanml/analysis/clustering.py
"""
Input cluster coverage analysis module.

Analyzes whether test data falls within the same input space as training data
using clustering techniques.

Example:
    from tanml.analysis.clustering import analyze_cluster_coverage
    
    coverage = analyze_cluster_coverage(
        X_train=train_features,
        X_test=test_features,
        n_clusters=5,
    )
    
    print(f"Coverage: {coverage['coverage_pct']:.1f}%")
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def analyze_cluster_coverage(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    n_clusters: int = 5,
    max_k: int = 10,
    auto_select_k: bool = False,
) -> Dict[str, Any]:
    """
    Analyze how well test data is covered by training data clusters.
    
    This check identifies whether test samples fall into regions of
    the input space that were seen during training.
    
    Args:
        X_train: Training features
        X_test: Test features
        n_clusters: Number of clusters (if auto_select_k=False)
        max_k: Maximum clusters to try (if auto_select_k=True)
        auto_select_k: Whether to auto-select optimal k using elbow method
        
    Returns:
        Dictionary with:
            - coverage_pct: Percentage of test samples in training clusters
            - cluster_distribution: Test samples per cluster
            - uncovered_indices: Indices of uncovered test samples
            - n_clusters: Actual number of clusters used
            - pca_coords: 2D PCA coordinates for visualization
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    
    # Get common numeric columns
    numeric_train = X_train.select_dtypes(include=[np.number])
    numeric_test = X_test.select_dtypes(include=[np.number])
    common_cols = list(set(numeric_train.columns) & set(numeric_test.columns))
    
    if not common_cols:
        return {
            "coverage_pct": 0.0,
            "cluster_distribution": {},
            "uncovered_indices": [],
            "n_clusters": 0,
            "error": "No common numeric columns found",
        }
    
    X_train_subset = numeric_train[common_cols].dropna()
    X_test_subset = numeric_test[common_cols].dropna()
    
    if len(X_train_subset) < n_clusters or len(X_test_subset) == 0:
        return {
            "coverage_pct": 0.0,
            "cluster_distribution": {},
            "uncovered_indices": list(range(len(X_test_subset))),
            "n_clusters": 0,
            "error": "Insufficient data for clustering",
        }
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_subset)
    X_test_scaled = scaler.transform(X_test_subset)
    
    # Auto-select k using elbow method if requested
    if auto_select_k:
        n_clusters = _select_optimal_k(X_train_scaled, max_k)
    
    # Fit KMeans on training data
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    train_labels = kmeans.fit_predict(X_train_scaled)
    test_labels = kmeans.predict(X_test_scaled)
    
    # Calculate distances to nearest cluster center
    train_distances = kmeans.transform(X_train_scaled).min(axis=1)
    test_distances = kmeans.transform(X_test_scaled).min(axis=1)
    
    # Define coverage threshold as max training distance (with buffer)
    threshold = np.percentile(train_distances, 95) * 1.5
    
    # Identify uncovered test samples
    uncovered_mask = test_distances > threshold
    uncovered_indices = np.where(uncovered_mask)[0].tolist()
    
    coverage_pct = 100 * (1 - uncovered_mask.mean())
    
    # Cluster distribution
    cluster_dist = {}
    for i in range(n_clusters):
        train_count = (train_labels == i).sum()
        test_count = (test_labels == i).sum()
        cluster_dist[i] = {
            "train_count": int(train_count),
            "test_count": int(test_count),
            "train_pct": float(100 * train_count / len(train_labels)),
            "test_pct": float(100 * test_count / len(test_labels)),
        }
    
    # PCA for visualization
    pca = PCA(n_components=2)
    train_pca = pca.fit_transform(X_train_scaled)
    test_pca = pca.transform(X_test_scaled)
    
    return {
        "coverage_pct": float(coverage_pct),
        "cluster_distribution": cluster_dist,
        "uncovered_indices": uncovered_indices,
        "uncovered_count": len(uncovered_indices),
        "n_clusters": n_clusters,
        "train_labels": train_labels.tolist(),
        "test_labels": test_labels.tolist(),
        "train_pca": train_pca.tolist(),
        "test_pca": test_pca.tolist(),
        "cluster_centers_pca": pca.transform(kmeans.cluster_centers_).tolist(),
        "status": "pass" if coverage_pct >= 90 else ("warning" if coverage_pct >= 70 else "fail"),
    }


def _select_optimal_k(X: np.ndarray, max_k: int = 10) -> int:
    """Select optimal k using elbow method."""
    from sklearn.cluster import KMeans
    
    max_k = min(max_k, len(X) - 1)
    if max_k < 2:
        return 2
    
    inertias = []
    K = range(2, max_k + 1)
    
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
    
    # Simple elbow detection: find point of maximum curvature
    if len(inertias) < 3:
        return 2
    
    # Calculate second derivative
    diffs = np.diff(inertias)
    diffs2 = np.diff(diffs)
    
    # Find elbow (max of second derivative)
    elbow_idx = np.argmax(diffs2) + 2  # +2 because of double diff
    
    return max(2, min(elbow_idx + 2, max_k))  # +2 to convert index to k
