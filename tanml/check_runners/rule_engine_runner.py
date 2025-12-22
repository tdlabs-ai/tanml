# tanml/check_runners/rule_engine_runner.py
"""
Rule engine check runner.

This check evaluates custom business rules defined in configuration
and flags any violations.
"""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from tanml.check_runners.base_runner import BaseCheckRunner
from tanml.checks.rule_engine import RuleEngineCheck


class RuleEngineCheckRunner(BaseCheckRunner):
    """
    Runner for custom business rule evaluation.
    
    Evaluates user-defined rules from configuration and reports
    any violations or warnings.
    
    Configuration:
        rules: List of rule definitions
        
    Output:
        - rules_passed: Number of rules that passed
        - rules_failed: Number of rules that failed
        - violations: List of rule violations with details
    """
    
    @property
    def name(self) -> str:
        return "RuleEngineCheck"
    
    def execute(self) -> Dict[str, Any]:
        """
        Evaluate business rules.
        
        Returns:
            Dictionary containing rule evaluation results
        """
        check = RuleEngineCheck(
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

def RuleEngineCheckRunnerLegacy(
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
    """Legacy function interface for RuleEngineCheck."""
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
    
    runner = RuleEngineCheckRunner(context)
    return runner.run() or {}
