# tanml/check_runners/correlation_runner.py
"""
Correlation check runner for analyzing feature correlations.

This check computes Pearson and Spearman correlation matrices,
identifies highly correlated pairs, and generates heatmap visualizations.
"""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from tanml.check_runners.base_runner import BaseCheckRunner
from tanml.checks.correlation import CorrelationCheck


class CorrelationCheckRunner(BaseCheckRunner):
    """
    Runner for correlation analysis between features.
    
    Computes correlation matrices (Pearson/Spearman), identifies
    highly correlated feature pairs, and generates visualizations.
    
    Configuration Options:
        method: "pearson" or "spearman" (default: "pearson")
        high_corr_threshold: Threshold for flagging pairs (default: 0.8)
        heatmap_max_features_default: Default features in heatmap (default: 20)
        heatmap_max_features_limit: Max features allowed (default: 60)
        subset_strategy: "cluster" or "degree" (default: "cluster")
        sample_rows: Max rows for computation (default: 150000)
        seed: Random seed for sampling (default: 42)
        
    Output Artifacts:
        - pearson_corr.csv: Full Pearson correlation matrix
        - spearman_corr.csv: Full Spearman correlation matrix
        - correlation_top_pairs.csv: Highly correlated pairs
        - heatmap.png: Correlation heatmap visualization
    """
    
    @property
    def name(self) -> str:
        return "CorrelationCheck"
    
    def execute(self) -> Dict[str, Any]:
        """
        Run correlation analysis on the cleaned data.
        
        Returns:
            Dictionary containing correlation results, file paths, and summary
        """
        # Build configuration from various sources
        cfg: Dict[str, Any] = {
            "method": self.get_config_value("method", "pearson"),
            "high_corr_threshold": float(
                self.get_config_value("high_corr_threshold", 0.8)
            ),
            "heatmap_max_features_default": int(
                self.get_config_value("heatmap_max_features_default", 20)
            ),
            "heatmap_max_features_limit": int(
                self.get_config_value("heatmap_max_features_limit", 60)
            ),
            "subset_strategy": self.get_config_value("subset_strategy", "cluster"),
            "sample_rows": int(
                self.get_config_value("sample_rows", 150_000)
            ),
            "seed": int(self.get_config_value("seed", 42)),
            "save_csv": True,
            "save_fig": True,
            "appendix_csv_cap": self.get_config_value("appendix_csv_cap"),
        }
        
        # Get features only (exclude target column)
        df = self._get_features_only(self.context.cleaned_df)
        
        # Run the check
        check = CorrelationCheck(
            cleaned_data=df,
            cfg=cfg,
            output_dir=str(self.get_output_dir()),
        )
        
        return check.run()
    
    def _get_features_only(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract feature columns only (exclude target).
        
        The target is assumed to be the last column in the DataFrame.
        """
        if df is None or df.empty:
            return df
        if len(df.columns) >= 2:
            return df.iloc[:, :-1]
        return df


# =============================================================================
# Legacy Compatibility - Function interface for backward compatibility
# =============================================================================

def CorrelationCheckRunnerLegacy(
    model,
    X_train,
    X_test,
    y_train,
    y_test,
    config: Dict[str, Any],
    cleaned_df: pd.DataFrame,
    raw_df: pd.DataFrame | None = None,
    ctx=None,
) -> Dict[str, Any] | None:
    """
    Legacy function interface for CorrelationCheck.
    
    This function maintains backward compatibility with the old runner interface.
    New code should use CorrelationCheckRunner class directly.
    """
    from tanml.core.context import CheckContext
    
    context = CheckContext(
        model=model,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        config=config,
        cleaned_df=cleaned_df,
        raw_df=raw_df,
    )
    
    runner = CorrelationCheckRunner(context)
    return runner.run()
