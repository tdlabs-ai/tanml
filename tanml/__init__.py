# tanml/__init__.py
"""
TanML - Industrial-Grade Model Validation Framework

TanML provides comprehensive model validation, testing, and reporting
for machine learning models in production environments.

Quick Start:
    from tanml import ValidationEngine, ReportBuilder
    
    # Create validation engine
    engine = ValidationEngine(
        model=my_model,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        config=config,
        cleaned_data=df,
    )
    
    # Run all validation checks
    results = engine.run_all_checks()
    
    # Generate report
    builder = ReportBuilder(results, template_path, output_path)
    builder.build()

For Contributors:
    See tanml/check_runners/base_runner.py for creating custom checks.
    See tanml/engine/check_agent_registry.py for registering new checks.
"""

__version__ = "0.1.8"

# Core abstractions (for contributors)
from tanml.core import (
    CheckContext,
    ReportContext,
    CheckRunnerProtocol,
    TanMLError,
    CheckRunnerError,
    ReportGenerationError,
)

# Configuration
from tanml.config import settings, TanMLSettings

# Main components
from tanml.engine.core_engine_agent import ValidationEngine
from tanml.report.report_builder import ReportBuilder

# Check runner base class (for contributors)
from tanml.check_runners.base_runner import BaseCheckRunner

# Registry functions (for contributors)
from tanml.engine.check_agent_registry import (
    register_check,
    list_available_checks,
)

__all__ = [
    # Version
    "__version__",
    # Core abstractions
    "CheckContext",
    "ReportContext",
    "CheckRunnerProtocol",
    "TanMLError",
    "CheckRunnerError",
    "ReportGenerationError",
    # Configuration
    "settings",
    "TanMLSettings",
    # Main components
    "ValidationEngine",
    "ReportBuilder",
    # For contributors
    "BaseCheckRunner",
    "register_check",
    "list_available_checks",
]
