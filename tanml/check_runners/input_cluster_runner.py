# tanml/check_runners/input_cluster_runner.py
"""
Input cluster coverage check runner.

This check evaluates how well the test data covers the input space
defined by the training data using K-Means clustering.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from tanml.check_runners.base_runner import BaseCheckRunner
from tanml.checks.input_cluster import InputClusterCoverageCheck


class InputClusterCheckRunner(BaseCheckRunner):
    """
    Runner for input cluster coverage analysis.
    
    Uses K-Means clustering to identify regions in the training data
    and checks how well the test data covers these regions.
    
    Configuration Options:
        n_clusters: Number of clusters (default: auto)
        
    Output:
        - coverage_pct: Percentage of clusters covered by test data
        - ood_pct: Percentage of test samples that are out-of-distribution
        - cluster_summary: Detailed cluster statistics
    """
    
    @property
    def name(self) -> str:
        return "InputClusterCheck"
    
    def execute(self) -> Dict[str, Any]:
        """
        Run input cluster coverage analysis.
        
        Returns:
            Dictionary containing coverage statistics
        """
        # Get expected features from model
        expected_features = self._get_expected_features()
        
        check = InputClusterCoverageCheck(
            cleaned_df=self.context.cleaned_df,
            feature_names=expected_features,
            rule_config=self.context.config,
        )
        
        return check.run()
    
    def _get_expected_features(self) -> List[str]:
        """Get feature names from model or context."""
        model = self.context.model
        
        # Try to get from model
        if hasattr(model, "feature_names_in_"):
            return list(model.feature_names_in_)
        
        # Fall back to context
        if self.context.feature_columns:
            return self.context.feature_columns
        
        # Fall back to X_train columns
        if hasattr(self.context.X_train, "columns"):
            return list(self.context.X_train.columns)
        
        raise ValueError(
            "Model does not have 'feature_names_in_' attribute and "
            "no feature columns specified in context"
        )
    
    @property
    def enabled(self) -> bool:
        """Check if enabled (defaults to False for this check)."""
        cfg = self.context.config.get("InputClusterCoverageCheck", {})
        return bool(cfg.get("enabled", False))


# =============================================================================
# Legacy Compatibility
# =============================================================================

def run_input_cluster_check(
    model,
    X_train,
    X_test,
    y_train,
    y_test,
    rule_config: Dict[str, Any],
    cleaned_df: pd.DataFrame,
    expected_features: List[str],
    *args,
    **kwargs,
) -> Dict[str, Any] | None:
    """Legacy function interface for InputClusterCheck."""
    from tanml.core.context import CheckContext
    
    context = CheckContext(
        model=model,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        config=rule_config,
        cleaned_df=cleaned_df,
        feature_columns=expected_features,
    )
    
    runner = InputClusterCheckRunner(context)
    return runner.run()
