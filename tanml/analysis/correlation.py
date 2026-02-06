# tanml/analysis/correlation.py
"""
Feature correlation and VIF analysis module.

Provides correlation matrix calculation and Variance Inflation Factor (VIF)
for detecting multicollinearity.

Example:
    from tanml.analysis.correlation import calculate_vif, calculate_correlation_matrix

    corr_matrix = calculate_correlation_matrix(df, method="pearson")
    vif_results = calculate_vif(df, features=["age", "income", "score"])
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def calculate_correlation_matrix(
    df: pd.DataFrame,
    method: str = "pearson",
    numeric_only: bool = True,
) -> pd.DataFrame:
    """
    Calculate correlation matrix for numeric features.

    Args:
        df: Input DataFrame
        method: Correlation method ("pearson", "spearman", or "kendall")
        numeric_only: Whether to include only numeric columns

    Returns:
        Correlation matrix as DataFrame
    """
    if numeric_only:
        df = df.select_dtypes(include=[np.number])

    return df.corr(method=method)


def find_highly_correlated_pairs(
    corr_matrix: pd.DataFrame,
    threshold: float = 0.8,
) -> list[dict[str, Any]]:
    """
    Find pairs of features with high correlation.

    Args:
        corr_matrix: Correlation matrix
        threshold: Absolute correlation threshold

    Returns:
        List of dictionaries with correlated pairs
    """
    pairs = []
    cols = corr_matrix.columns.tolist()

    for i, col1 in enumerate(cols):
        for j, col2 in enumerate(cols):
            if i < j:  # Upper triangle only
                corr = corr_matrix.loc[col1, col2]
                if abs(corr) >= threshold:
                    pairs.append(
                        {
                            "feature_1": col1,
                            "feature_2": col2,
                            "correlation": float(corr),
                        }
                    )

    # Sort by absolute correlation (highest first)
    pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)
    return pairs


def calculate_vif(
    df: pd.DataFrame,
    features: list[str] | None = None,
    threshold: float = 5.0,
) -> dict[str, Any]:
    """
    Calculate Variance Inflation Factor (VIF) for features.

    VIF measures how much the variance of a regression coefficient is
    inflated due to multicollinearity. Thresholds:
        - VIF < 5: Low multicollinearity
        - 5 <= VIF < 10: Moderate multicollinearity
        - VIF >= 10: High multicollinearity

    Args:
        df: Input DataFrame
        features: List of features to analyze (auto-detected if None)
        threshold: VIF threshold for flagging

    Returns:
        Dictionary with VIF values and flagged features
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    # Get numeric columns
    if features is None:
        features = df.select_dtypes(include=[np.number]).columns.tolist()

    # Filter to existing columns
    features = [f for f in features if f in df.columns]

    if len(features) < 2:
        return {
            "vif_values": {},
            "high_vif_features": [],
            "status": "pass",
            "error": "Need at least 2 features for VIF calculation",
        }

    # Prepare data (drop NaN rows)
    X = df[features].dropna()

    if len(X) < len(features) + 1:
        return {
            "vif_values": {},
            "high_vif_features": [],
            "status": "unknown",
            "error": "Insufficient data for VIF calculation",
        }

    # Calculate VIF for each feature
    vif_values = {}
    for i, col in enumerate(features):
        try:
            vif = variance_inflation_factor(X.values, i)
            vif_values[col] = float(vif) if not np.isinf(vif) else float("inf")
        except Exception:
            vif_values[col] = np.nan

    # Find high VIF features
    high_vif = [
        col
        for col, vif in vif_values.items()
        if vif is not None and not np.isnan(vif) and vif >= threshold
    ]

    return {
        "vif_values": vif_values,
        "high_vif_features": high_vif,
        "high_vif_count": len(high_vif),
        "threshold": threshold,
        "status": "warning" if high_vif else "pass",
    }


def analyze_feature_relationships(
    df: pd.DataFrame,
    features: list[str] | None = None,
    corr_method: str = "pearson",
    corr_threshold: float = 0.8,
    vif_threshold: float = 5.0,
) -> dict[str, Any]:
    """
    Comprehensive feature relationship analysis.

    Combines correlation and VIF analysis for a complete picture
    of feature relationships.

    Args:
        df: Input DataFrame
        features: List of features to analyze
        corr_method: Correlation method
        corr_threshold: Threshold for flagging high correlations
        vif_threshold: Threshold for flagging high VIF

    Returns:
        Combined analysis results
    """
    if features is None:
        features = df.select_dtypes(include=[np.number]).columns.tolist()

    df_subset = df[features]

    # Correlation analysis
    corr_matrix = calculate_correlation_matrix(df_subset, method=corr_method)
    high_corr_pairs = find_highly_correlated_pairs(corr_matrix, threshold=corr_threshold)

    # VIF analysis
    vif_results = calculate_vif(df_subset, features=features, threshold=vif_threshold)

    # Determine overall status
    has_high_corr = len(high_corr_pairs) > 0
    has_high_vif = len(vif_results.get("high_vif_features", [])) > 0

    if has_high_corr or has_high_vif:
        status = "warning"
    else:
        status = "pass"

    return {
        "correlation_matrix": corr_matrix.to_dict(),
        "high_correlation_pairs": high_corr_pairs,
        "vif": vif_results,
        "status": status,
        "summary": {
            "n_features": len(features),
            "n_high_corr_pairs": len(high_corr_pairs),
            "n_high_vif_features": len(vif_results.get("high_vif_features", [])),
        },
    }
