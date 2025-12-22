# tanml/check_runners/model_meta_runner.py
"""
Model metadata check runner.

This check extracts and validates model metadata including:
- Model type and algorithm
- Feature names and count
- Target variable statistics
- Training parameters
"""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from tanml.check_runners.base_runner import BaseCheckRunner
from tanml.checks.model_meta import ModelMetaCheck


class ModelMetaCheckRunner(BaseCheckRunner):
    """
    Runner for model metadata extraction.
    
    Collects comprehensive metadata about the model and data
    for report generation and validation.
    
    Output:
        - model_type: Type of model (e.g., "RandomForestClassifier")
        - feature_count: Number of features
        - feature_names: List of feature names
        - target_balance: Class distribution (classification) or statistics (regression)
        - sample_sizes: Train/test sample counts
    """
    
    @property
    def name(self) -> str:
        return "ModelMetaCheck"
    
    def execute(self) -> Dict[str, Any]:
        """
        Extract model metadata.
        
        Returns:
            Dictionary containing model and data metadata
        """
        check = ModelMetaCheck(
            model=self.context.model,
            X_train=self.context.X_train,
            X_test=self.context.X_test,
            y_train=self.context.y_train,
            y_test=self.context.y_test,
            rule_config=self.context.config,
            cleaned_data=self.context.cleaned_df,
        )
        
        return check.run()


# =============================================================================
# Legacy Compatibility
# =============================================================================

def ModelMetaCheckRunnerLegacy(
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
    """Legacy function interface for ModelMetaCheck."""
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
    
    runner = ModelMetaCheckRunner(context)
    return runner.run()
