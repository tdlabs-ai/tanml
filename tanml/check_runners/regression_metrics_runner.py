# tanml/check_runners/regression_metrics_runner.py
"""
Regression metrics check runner.

This check computes regression-specific performance metrics including:
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- Median Absolute Error
- R² Score
"""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from tanml.check_runners.base_runner import BaseCheckRunner
from tanml.checks.regression_metrics import RegressionMetricsCheck


class RegressionMetricsCheckRunner(BaseCheckRunner):
    """
    Runner for regression model performance metrics.
    
    Computes comprehensive regression metrics and generates
    diagnostic visualizations (residual plots, QQ plots, etc.).
    
    Output:
        - rmse: Root Mean Squared Error
        - mae: Mean Absolute Error
        - median_ae: Median Absolute Error
        - r2: R² Score
        - plots: Diagnostic visualizations
    """
    
    @property
    def name(self) -> str:
        return "RegressionMetricsCheck"
    
    def execute(self) -> Dict[str, Any]:
        """
        Compute regression metrics.
        
        Returns:
            Dictionary containing regression performance metrics
        """
        output_dir = str(self.get_output_dir())
        
        check = RegressionMetricsCheck(
            model=self.context.model,
            X_train=self.context.X_train,
            X_test=self.context.X_test,
            y_train=self.context.y_train,
            y_test=self.context.y_test,
            rule_config=self.context.config,
            cleaned_data=self.context.cleaned_df,
            output_dir=output_dir,
        )
        
        return check.run()


# =============================================================================
# Legacy Compatibility
# =============================================================================

def RegressionMetricsCheckRunnerLegacy(
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
    """Legacy function interface for RegressionMetricsCheck."""
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
    
    runner = RegressionMetricsCheckRunner(context)
    return runner.run()
