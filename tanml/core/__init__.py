# tanml/core/__init__.py
"""
Core abstractions for the TanML validation framework.

This module provides the foundational types and protocols that enable:
- Consistent interfaces for check runners
- Type-safe dependency injection
- Plugin-friendly architecture for contributors

Example usage for contributors adding new checks:
    
    from tanml.core import CheckContext, BaseCheckRunner
    
    class MyCustomCheck(BaseCheckRunner):
        @property
        def name(self) -> str:
            return "MyCustomCheck"
        
        def execute(self) -> Dict[str, Any]:
            # Your check logic here
            return {"status": "ok"}
"""

from tanml.core.context import CheckContext, ReportContext
from tanml.core.protocols import (
    CheckRunnerProtocol,
    ReportSectionBuilderProtocol,
)
from tanml.core.exceptions import (
    TanMLError,
    CheckRunnerError,
    ReportGenerationError,
    ConfigurationError,
)

__all__ = [
    # Context objects
    "CheckContext",
    "ReportContext",
    # Protocols (interfaces)
    "CheckRunnerProtocol",
    "ReportSectionBuilderProtocol",
    # Exceptions
    "TanMLError",
    "CheckRunnerError",
    "ReportGenerationError",
    "ConfigurationError",
]
