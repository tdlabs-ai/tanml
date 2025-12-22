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

__all__ = [
    "__version__",
]

