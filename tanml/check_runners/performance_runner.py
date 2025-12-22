# tanml/check_runners/performance_runner.py
"""
Performance check runner for computing model metrics.

This check evaluates model performance using task-appropriate metrics:
- Classification: AUC, F1, Precision, Recall, Accuracy, etc.
- Regression: RMSE, MAE, R2, etc.
"""

from __future__ import annotations

import os
from typing import Any, Dict

import numpy as np
import pandas as pd

from tanml.check_runners.base_runner import BaseCheckRunner
from tanml.checks.performance_classification import compute_classification_report


class PerformanceCheckRunner(BaseCheckRunner):
    """
    Runner for model performance evaluation.
    
    Computes comprehensive performance metrics based on task type
    and generates diagnostic visualizations.
    
    Configuration:
        model.type: Task type ("classification" or "regression")
        model.name: Display name for the model
        
    Output:
        - performance.classification: Classification metrics and curves
        - performance.regression: Regression metrics
        - task_type: Detected/configured task type
    """
    
    @property
    def name(self) -> str:
        return "PerformanceCheck"
    
    def execute(self) -> Dict[str, Any]:
        """
        Compute performance metrics for the model.
        
        Returns:
            Dictionary containing performance metrics and file paths
        """
        # Get task type from config
        model_config = self.context.config.get("model", {})
        task_type = (model_config.get("type") or "binary_classification").lower()
        
        payload: Dict[str, Any] = {}
        
        if "class" in task_type:
            payload = self._run_classification()
        else:
            payload = self._run_regression()
        
        return payload
    
    def _run_classification(self) -> Dict[str, Any]:
        """Compute classification metrics."""
        model = self.context.model
        X_test = self.context.X_test
        y_test = self.context.y_test
        
        # Get prediction scores
        y_score = self._get_scores(model, X_test)
        y_pred = model.predict(X_test)
        
        # Get output directory
        cls_dir = str(self.get_output_dir("classification"))
        
        # Get model name for plot titles
        model_name = (
            self.context.config.get("model", {}).get("name", "Model")
        )
        
        # Compute classification report
        results_cls = compute_classification_report(
            y_true=np.asarray(y_test),
            y_score=np.asarray(y_score),
            y_pred=np.asarray(y_pred),
            outdir=cls_dir,
            pos_label=1,
            title_prefix=model_name,
        )
        
        return {
            "performance": {"classification": results_cls},
            "task_type": "classification",
        }
    
    def _run_regression(self) -> Dict[str, Any]:
        """Compute regression metrics."""
        # Regression metrics are handled by RegressionMetricsCheckRunner
        return {"task_type": "regression"}
    
    def _get_scores(self, model, X) -> np.ndarray:
        """Get prediction scores from model."""
        if hasattr(model, "predict_proba"):
            p = model.predict_proba(X)
            return p[:, 1] if p.ndim == 2 and p.shape[1] > 1 else p.ravel()
        if hasattr(model, "decision_function"):
            return model.decision_function(X).ravel()
        return model.predict(X).ravel()


# =============================================================================
# Legacy Compatibility
# =============================================================================

def run_performance_check(
    model,
    X_train,
    X_test,
    y_train,
    y_test,
    config: Dict[str, Any],
    cleaned_df,
    raw_df=None,
    ctx=None,
) -> Dict[str, Any]:
    """Legacy function interface for PerformanceCheck."""
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
    
    runner = PerformanceCheckRunner(context)
    return runner.run() or {}
