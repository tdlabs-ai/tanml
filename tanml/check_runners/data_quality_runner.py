# tanml/check_runners/data_quality_runner.py
"""
Data quality check runner.

This check evaluates data quality metrics including:
- Missing value patterns
- Data type consistency
- Value range validation
"""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from tanml.check_runners.base_runner import BaseCheckRunner
from tanml.checks.data_quality import DataQualityCheck


class DataQualityCheckRunner(BaseCheckRunner):
    """
    Runner for data quality assessment.
    
    Analyzes data quality metrics for both training and test data
    to identify potential issues.
    
    Output:
        - missing_summary: Missing value statistics
        - type_summary: Data type distribution
        - quality_score: Overall quality assessment
    """
    
    @property
    def name(self) -> str:
        return "DataQualityCheck"
    
    def execute(self) -> Dict[str, Any]:
        """
        Run data quality analysis.
        
        Returns:
            Dictionary containing data quality metrics
        """
        check = DataQualityCheck(
            model=self.context.model,
            X_train=self.context.X_train,
            X_test=self.context.X_test,
            y_train=self.context.y_train,
            y_test=self.context.y_test,
            rule_config=self._config,
            cleaned_data=self.context.cleaned_df,
        )
        
        result = check.run()
        return {"DataQualityCheck": result}


# =============================================================================
# Legacy Compatibility
# =============================================================================

def run_data_quality_check(
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
    """Legacy function interface for DataQualityCheck."""
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
    
    runner = DataQualityCheckRunner(context)
    return runner.run() or {"DataQualityCheck": {"skipped": True}}
