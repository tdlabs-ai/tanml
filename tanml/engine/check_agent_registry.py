# tanml/engine/check_agent_registry.py
"""
Check runner registry for the validation engine.

This module provides the central registry of all available check runners.
The engine uses this registry to discover and instantiate checks.

For Contributors - Adding a New Check:
    1. Create your check runner in tanml/check_runners/my_check_runner.py
    2. Inherit from BaseCheckRunner
    3. Import and register it here in CHECK_RUNNER_REGISTRY
    
    Example:
        from tanml.check_runners.my_check_runner import MyCheckRunner
        
        CHECK_RUNNER_REGISTRY["MyCheck"] = MyCheckRunner
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Type, Union

# Import runner classes (new style)
from tanml.check_runners.correlation_runner import (
    CorrelationCheckRunner,
    CorrelationCheckRunnerLegacy,
)
from tanml.check_runners.eda_runner import (
    EDACheckRunner,
    EDACheckRunnerLegacy,
)
from tanml.check_runners.vif_runner import (
    VIFCheckRunner,
    VIFCheckRunnerLegacy,
)
from tanml.check_runners.performance_runner import (
    PerformanceCheckRunner,
    run_performance_check,
)
from tanml.check_runners.model_meta_runner import (
    ModelMetaCheckRunner,
    ModelMetaCheckRunnerLegacy,
)
from tanml.check_runners.stress_test_runner import (
    StressTestCheckRunner,
    run_stress_test_check,
)
from tanml.check_runners.regression_metrics_runner import (
    RegressionMetricsCheckRunner,
    RegressionMetricsCheckRunnerLegacy,
)
from tanml.check_runners.input_cluster_runner import (
    InputClusterCheckRunner,
    run_input_cluster_check,
)
from tanml.check_runners.explainability_runner import (
    SHAPCheckRunner,
    run_shap_check,
)
from tanml.check_runners.data_quality_runner import (
    DataQualityCheckRunner,
    run_data_quality_check,
)
from tanml.check_runners.raw_data_runner import (
    RawDataCheckRunner,
    run_raw_data_check,
)
from tanml.check_runners.logistic_stats_runner import (
    LogisticStatsCheckRunner,
    run_logistic_stats_check,
)
from tanml.check_runners.rule_engine_runner import (
    RuleEngineCheckRunner,
    RuleEngineCheckRunnerLegacy,
)


# =============================================================================
# Input Cluster Wrapper (for backward compatibility)
# =============================================================================

def input_cluster_wrapper(model, X_train, X_test, y_train, y_test, rule_config, cleaned_df, *args, **kwargs):
    """Wrapper to inject expected_features from model."""
    if hasattr(model, "feature_names_in_"):
        expected_features = list(model.feature_names_in_)
    else:
        raise ValueError("Model does not have 'feature_names_in_' attribute required for InputClusterCheck.")
    
    return run_input_cluster_check(
        model, X_train, X_test, y_train, y_test, rule_config, cleaned_df, expected_features
    )


# =============================================================================
# Check Runner Registry
# =============================================================================

# Legacy registry (function-based) - maintained for backward compatibility
CHECK_RUNNER_REGISTRY: Dict[str, Callable[..., Dict[str, Any] | None]] = {
    "RawDataCheck": run_raw_data_check,
    "DataQualityCheck": run_data_quality_check,
    "EDACheck": EDACheckRunnerLegacy,
    "CorrelationCheck": CorrelationCheckRunnerLegacy,
    "VIFCheck": VIFCheckRunnerLegacy,
    "InputClusterCheck": input_cluster_wrapper,
    "ModelMetaCheck": ModelMetaCheckRunnerLegacy,
    "PerformanceCheck": run_performance_check,
    "RegressionMetricsCheck": RegressionMetricsCheckRunnerLegacy,
    "LogisticStatsCheck": run_logistic_stats_check,
    "StressTestCheck": run_stress_test_check,
    "SHAPCheck": run_shap_check,
    "RuleEngineCheck": RuleEngineCheckRunnerLegacy,
}


# New class-based registry - for modern usage
CHECK_RUNNER_CLASSES: Dict[str, Type] = {
    "RawDataCheck": RawDataCheckRunner,
    "DataQualityCheck": DataQualityCheckRunner,
    "EDACheck": EDACheckRunner,
    "CorrelationCheck": CorrelationCheckRunner,
    "VIFCheck": VIFCheckRunner,
    "InputClusterCheck": InputClusterCheckRunner,
    "ModelMetaCheck": ModelMetaCheckRunner,
    "PerformanceCheck": PerformanceCheckRunner,
    "RegressionMetricsCheck": RegressionMetricsCheckRunner,
    "LogisticStatsCheck": LogisticStatsCheckRunner,
    "StressTestCheck": StressTestCheckRunner,
    "SHAPCheck": SHAPCheckRunner,
    "RuleEngineCheck": RuleEngineCheckRunner,
}


def get_runner_class(check_name: str) -> Type | None:
    """
    Get the runner class for a check name.
    
    Args:
        check_name: Name of the check (e.g., "CorrelationCheck")
        
    Returns:
        The runner class, or None if not found
    """
    return CHECK_RUNNER_CLASSES.get(check_name)


def list_available_checks() -> list[str]:
    """
    List all available check names.
    
    Returns:
        List of check names that can be run
    """
    return list(CHECK_RUNNER_CLASSES.keys())


def register_check(name: str, runner_class: Type, legacy_func: Callable = None) -> None:
    """
    Register a new check runner.
    
    This is the preferred way for contributors to add new checks.
    
    Args:
        name: Unique check name (e.g., "MyCustomCheck")
        runner_class: The runner class (must inherit from BaseCheckRunner)
        legacy_func: Optional legacy function for backward compatibility
        
    Example:
        from tanml.engine.check_agent_registry import register_check
        from my_module import MyCustomCheckRunner, my_custom_check_legacy
        
        register_check("MyCustomCheck", MyCustomCheckRunner, my_custom_check_legacy)
    """
    CHECK_RUNNER_CLASSES[name] = runner_class
    if legacy_func:
        CHECK_RUNNER_REGISTRY[name] = legacy_func
