# tanml/check_runners/stress_test_runner.py
"""
Stress test check runner.

This check evaluates model robustness by introducing noise
to the test data and measuring performance degradation.
"""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from tanml.check_runners.base_runner import BaseCheckRunner
from tanml.checks.stress_test import StressTestCheck


class StressTestCheckRunner(BaseCheckRunner):
    """
    Runner for model stress testing.
    
    Perturbs the test data with noise and compares performance
    against the baseline to assess model robustness.
    
    Configuration Options:
        noise_fraction: Fraction of data to perturb (default: 0.2)
        seed: Random seed for reproducibility
        
    Output:
        - baseline_metrics: Performance on original data
        - stressed_metrics: Performance on perturbed data
        - delta_*: Performance differences
    """
    
    @property
    def name(self) -> str:
        return "StressTestCheck"
    
    def execute(self) -> Dict[str, Any]:
        """
        Run stress test analysis.
        
        Returns:
            Dictionary containing stress test results
        """
        seed = self.get_config_value("seed", 42)
        
        check = StressTestCheck(
            model=self.context.model,
            X_train=self.context.X_train,
            X_test=self.context.X_test,
            y_train=self.context.y_train,
            y_test=self.context.y_test,
            rule_config=self.context.config,
            cleaned_data=self.context.cleaned_df,
            seed=seed,
        )
        
        return check.run()


# =============================================================================
# Legacy Compatibility
# =============================================================================

def run_stress_test_check(
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
    """Legacy function interface for StressTestCheck."""
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
    
    runner = StressTestCheckRunner(context)
    return runner.run()
