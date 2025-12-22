# tanml/check_runners/__init__.py
"""
Check runners for TanML validation engine.

This package contains all validation check runners. Each runner implements
the BaseCheckRunner interface to ensure consistency.

For Contributors - Adding a New Check:
    1. Create a new file: tanml/check_runners/my_check_runner.py
    2. Inherit from BaseCheckRunner
    3. Implement `name` property and `execute()` method
    4. Register in CHECK_RUNNER_REGISTRY (check_agent_registry.py)
    
    Example:
        from tanml.check_runners.base_runner import BaseCheckRunner
        
        class MyCheckRunner(BaseCheckRunner):
            @property
            def name(self) -> str:
                return "MyCheck"
            
            def execute(self) -> Dict[str, Any]:
                return {"status": "ok"}

Available Runners:
    - CorrelationCheckRunner: Analyzes feature correlations
    - EDACheckRunner: Exploratory data analysis
    - VIFCheckRunner: Variance inflation factor analysis
    - PerformanceCheckRunner: Model performance metrics
    - StressTestCheckRunner: Robustness testing
    - SHAPCheckRunner: SHAP explainability
    - And more...
"""

from tanml.check_runners.base_runner import BaseCheckRunner, create_runner_from_function

__all__ = [
    "BaseCheckRunner",
    "create_runner_from_function",
]
