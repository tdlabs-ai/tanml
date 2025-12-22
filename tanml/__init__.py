# tanml/__init__.py
"""
TanML - Industrial-Grade Model Validation Framework

TanML provides comprehensive model validation, testing, and reporting
for machine learning models in production environments.

Quick Start:
    # Launch the UI
    tanml ui
    
    # Or run checks programmatically
    from tanml.checks.stress_test import StressTestCheck
    from tanml.checks.explainability.shap_check import SHAPCheck
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
]
