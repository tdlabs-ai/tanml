# tanml/check_runners/vif_runner.py
"""
VIF (Variance Inflation Factor) check runner.

This check detects multicollinearity between features by computing
VIF scores. High VIF values indicate features that are highly correlated
with other features and may cause instability in linear models.
"""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from tanml.check_runners.base_runner import BaseCheckRunner
from tanml.checks.vif import VIFCheck


class VIFCheckRunner(BaseCheckRunner):
    """
    Runner for VIF multicollinearity analysis.
    
    Computes Variance Inflation Factor for each feature to detect
    multicollinearity issues in the dataset.
    
    Configuration Options:
        threshold: VIF threshold for flagging features (default: 5.0)
        
    Output:
        - vif_scores: Dictionary of feature -> VIF score
        - high_vif_features: Features with VIF above threshold
        - summary: Overall multicollinearity assessment
    """
    
    @property
    def name(self) -> str:
        return "VIFCheck"
    
    def execute(self) -> Dict[str, Any]:
        """
        Compute VIF scores for all features.
        
        Returns:
            Dictionary containing VIF scores and analysis
        """
        threshold = float(self.get_config_value("threshold", 5.0))
        
        # Get output directory
        output_dir = str(self.get_output_dir())
        
        check = VIFCheck(
            cleaned_data=self.context.cleaned_df,
            threshold=threshold,
            output_dir=output_dir,
        )
        
        return check.run()


# =============================================================================
# Legacy Compatibility
# =============================================================================

def VIFCheckRunnerLegacy(
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
    """Legacy function interface for VIFCheck."""
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
    
    runner = VIFCheckRunner(context)
    return runner.run()
