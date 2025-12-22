# tanml/analysis/drift.py
"""
Feature drift analysis module.

Provides PSI (Population Stability Index) and KS (Kolmogorov-Smirnov) 
statistics for detecting distribution shifts between train and test data.

Example:
    from tanml.analysis.drift import analyze_drift
    
    drift_results = analyze_drift(
        train_df=X_train,
        test_df=X_test,
        numeric_cols=["age", "income", "score"],
    )
    
    for col, metrics in drift_results.items():
        print(f"{col}: PSI={metrics['psi']:.3f}, KS={metrics['ks']:.3f}")
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd


def calculate_psi(
    expected: pd.Series,
    actual: pd.Series,
    bins: int = 10,
) -> float:
    """
    Calculate Population Stability Index (PSI) between two distributions.
    
    PSI measures how much a distribution has shifted. Thresholds:
        - PSI < 0.1: No significant shift
        - 0.1 <= PSI < 0.2: Moderate shift (investigate)
        - PSI >= 0.2: Large shift (action needed)
    
    Args:
        expected: Expected/baseline distribution (e.g., training data)
        actual: Actual/new distribution (e.g., test data)
        bins: Number of bins for discretization
        
    Returns:
        PSI value (float)
    """
    # Handle edge cases
    expected = expected.dropna()
    actual = actual.dropna()
    
    if len(expected) == 0 or len(actual) == 0:
        return np.nan
    
    # Create bins from expected distribution
    try:
        _, bin_edges = np.histogram(expected, bins=bins)
    except ValueError:
        return np.nan
    
    # Calculate proportions in each bin
    expected_counts = np.histogram(expected, bins=bin_edges)[0]
    actual_counts = np.histogram(actual, bins=bin_edges)[0]
    
    # Convert to proportions (avoid division by zero)
    expected_pct = expected_counts / len(expected)
    actual_pct = actual_counts / len(actual)
    
    # Replace zeros with small value to avoid log(0)
    eps = 1e-8
    expected_pct = np.where(expected_pct == 0, eps, expected_pct)
    actual_pct = np.where(actual_pct == 0, eps, actual_pct)
    
    # Calculate PSI
    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    
    return float(psi)


def calculate_ks(
    expected: pd.Series,
    actual: pd.Series,
) -> Tuple[float, float]:
    """
    Calculate Kolmogorov-Smirnov statistic between two distributions.
    
    Args:
        expected: Expected/baseline distribution
        actual: Actual/new distribution
        
    Returns:
        Tuple of (KS statistic, p-value)
    """
    from scipy import stats
    
    expected = expected.dropna()
    actual = actual.dropna()
    
    if len(expected) == 0 or len(actual) == 0:
        return np.nan, np.nan
    
    try:
        ks_stat, p_value = stats.ks_2samp(expected, actual)
        return float(ks_stat), float(p_value)
    except Exception:
        return np.nan, np.nan


def analyze_drift(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    numeric_cols: Optional[List[str]] = None,
    psi_threshold: float = 0.1,
    ks_threshold: float = 0.05,
) -> Dict[str, Dict[str, Any]]:
    """
    Analyze feature drift between training and test datasets.
    
    Args:
        train_df: Training dataset
        test_df: Test dataset  
        numeric_cols: List of numeric columns to analyze (auto-detected if None)
        psi_threshold: PSI threshold for flagging drift
        ks_threshold: KS p-value threshold for flagging drift
        
    Returns:
        Dictionary with drift metrics for each column:
        {
            "column_name": {
                "psi": float,
                "ks_statistic": float,
                "ks_pvalue": float,
                "has_drift": bool,
                "drift_level": "none" | "moderate" | "severe"
            }
        }
    """
    if numeric_cols is None:
        numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Get common columns
    common_cols = [c for c in numeric_cols if c in train_df.columns and c in test_df.columns]
    
    results = {}
    for col in common_cols:
        psi = calculate_psi(train_df[col], test_df[col])
        ks_stat, ks_pval = calculate_ks(train_df[col], test_df[col])
        
        # Determine drift level
        if np.isnan(psi):
            drift_level = "unknown"
            has_drift = False
        elif psi >= 0.2:
            drift_level = "severe"
            has_drift = True
        elif psi >= psi_threshold:
            drift_level = "moderate"
            has_drift = True
        else:
            drift_level = "none"
            has_drift = False
        
        results[col] = {
            "psi": psi,
            "ks_statistic": ks_stat,
            "ks_pvalue": ks_pval,
            "has_drift": has_drift,
            "drift_level": drift_level,
        }
    
    return results


def get_drift_summary(drift_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Summarize drift analysis results.
    
    Args:
        drift_results: Output from analyze_drift()
        
    Returns:
        Summary dictionary with counts and lists of drifted features
    """
    severe = [col for col, metrics in drift_results.items() if metrics["drift_level"] == "severe"]
    moderate = [col for col, metrics in drift_results.items() if metrics["drift_level"] == "moderate"]
    
    return {
        "total_features": len(drift_results),
        "severe_drift_count": len(severe),
        "moderate_drift_count": len(moderate),
        "severe_drift_features": severe,
        "moderate_drift_features": moderate,
        "overall_status": "fail" if severe else ("warning" if moderate else "pass"),
    }
