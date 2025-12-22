# tanml/check_runners/eda_runner.py
"""
EDA (Exploratory Data Analysis) check runner.

This check performs comprehensive data exploration including:
- Data health metrics (missing values, duplicates)
- Target variable analysis
- PCA projections
- Outlier detection
- Feature distributions
"""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from tanml.check_runners.base_runner import BaseCheckRunner
from tanml.checks.eda import EDACheck


class EDACheckRunner(BaseCheckRunner):
    """
    Runner for Exploratory Data Analysis.
    
    Performs comprehensive data exploration and generates
    visualizations for understanding data characteristics.
    
    Configuration Options:
        max_plots: Maximum number of distribution plots (-1 for unlimited)
        
    Output Artifacts:
        - summary_stats.csv: Descriptive statistics
        - missing_values.csv: Missing value analysis
        - nullity_matrix.png: Missing values heatmap
        - target_analysis.png: Target variable distribution
        - pca_projection.png: 2D PCA visualization
        - outlier_scores.png: Anomaly score distribution
        - dist_*.png: Individual feature distributions
    """
    
    @property
    def name(self) -> str:
        return "EDACheck"
    
    def execute(self) -> Dict[str, Any]:
        """
        Run exploratory data analysis.
        
        Returns:
            Dictionary containing EDA results, file paths, and health metrics
        """
        # Create the check with all required data
        check = EDACheck(
            cleaned_data=self.context.cleaned_df,
            model=self.context.model,
            X_train=self.context.X_train,
            X_test=self.context.X_test,
            y_train=self.context.y_train,
            y_test=self.context.y_test,
            rule_config=self.context.config,
            output_dir=str(self.get_output_dir()),
        )
        
        return check.run()


# =============================================================================
# Legacy Compatibility
# =============================================================================

def EDACheckRunnerLegacy(
    model,
    X_train,
    X_test,
    y_train,
    y_test,
    rule_config: Dict[str, Any],
    cleaned_df: pd.DataFrame,
    *args,
    **kwargs,
) -> Dict[str, Any] | None:
    """
    Legacy function interface for EDACheck.
    
    Maintains backward compatibility with the old runner interface.
    """
    from tanml.core.context import CheckContext
    
    context = CheckContext(
        model=model,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        config=rule_config,
        cleaned_df=cleaned_df,
    )
    
    runner = EDACheckRunner(context)
    return runner.run()
