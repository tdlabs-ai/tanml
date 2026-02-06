# tanml/analysis/__init__.py
"""
Analysis module for TanML.

This module contains reusable business logic for data analysis,
separated from the UI layer for better maintainability and testability.

Modules:
    - drift: Feature drift analysis (PSI, KS statistics)
    - clustering: Input cluster coverage analysis
    - correlation: Feature correlation and VIF analysis
    - benchmarking: Model comparison and benchmarking

Example:
    from tanml.analysis import calculate_psi, calculate_ks

    psi_scores = calculate_psi(train_df, test_df, numeric_cols)
    ks_scores = calculate_ks(train_df, test_df, numeric_cols)
"""

from tanml.analysis.clustering import (
    analyze_cluster_coverage,
)
from tanml.analysis.correlation import (
    calculate_correlation_matrix,
    calculate_vif,
)
from tanml.analysis.drift import (
    analyze_drift,
    calculate_ks,
    calculate_psi,
)

__all__ = [
    # Clustering
    "analyze_cluster_coverage",
    "analyze_drift",
    # Correlation
    "calculate_correlation_matrix",
    "calculate_ks",
    # Drift analysis
    "calculate_psi",
    "calculate_vif",
]
