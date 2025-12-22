# tanml/check_runners/explainability_runner.py
"""
SHAP explainability check runner.

This check uses SHAP (SHapley Additive exPlanations) to explain
model predictions and identify the most influential features.
"""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from tanml.check_runners.base_runner import BaseCheckRunner
from tanml.checks.explainability.shap_check import SHAPCheck


class SHAPCheckRunner(BaseCheckRunner):
    """
    Runner for SHAP explainability analysis.
    
    Computes SHAP values to explain model predictions and
    generates visualizations (beeswarm, bar plots).
    
    Configuration Options:
        background_size: Number of background samples (default: 100)
        test_size: Number of test samples to explain (default: 200)
        
    Output:
        - top_features: Most influential features
        - plots: SHAP visualizations
    """
    
    @property
    def name(self) -> str:
        return "SHAPCheck"
    
    @property
    def enabled(self) -> bool:
        """Check if enabled from multiple config sources."""
        cfg = self.context.config.get("SHAPCheck", {})
        cfg_expl = (
            self.context.config.get("explainability", {})
            .get("shap", {})
        )
        return bool(cfg.get("enabled", cfg_expl.get("enabled", True)))
    
    def execute(self) -> Dict[str, Any]:
        """
        Run SHAP analysis.
        
        Returns:
            Dictionary containing SHAP results and visualizations
        """
        check = SHAPCheck(
            model=self.context.model,
            X_train=self.context.X_train,
            X_test=self.context.X_test,
            y_train=self.context.y_train,
            y_test=self.context.y_test,
            rule_config=self.context.config,
            cleaned_df=self.context.cleaned_df,
        )
        
        result = check.run()
        return {"SHAPCheck": result}


# =============================================================================
# Legacy Compatibility
# =============================================================================

def run_shap_check(
    model,
    X_train,
    X_test,
    y_train,
    y_test,
    rule_config: Dict[str, Any],
    cleaned_df: pd.DataFrame,
    *args,
    **kwargs,
) -> Dict[str, Any]:
    """Legacy function interface for SHAPCheck."""
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
    
    runner = SHAPCheckRunner(context)
    return runner.run() or {"SHAPCheck": {"skipped": True}}
