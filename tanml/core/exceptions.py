# tanml/core/exceptions.py
"""
Exception hierarchy for TanML.

A clear exception hierarchy makes error handling consistent and helps
contributors understand what can go wrong and how to handle it.

Exception Hierarchy:
    TanMLError (base)
    ├── CheckRunnerError
    │   ├── CheckNotFoundError
    │   ├── CheckExecutionError
    │   └── CheckConfigurationError
    ├── ReportGenerationError
    │   ├── TemplateNotFoundError
    │   └── SectionBuildError
    ├── ConfigurationError
    │   └── ValidationError
    └── DataLoadError
"""

from typing import Any, Dict, Optional


class TanMLError(Exception):
    """
    Base exception for all TanML errors.
    
    All custom exceptions in TanML inherit from this class,
    making it easy to catch all TanML-specific errors:
    
        try:
            engine.run_all_checks()
        except TanMLError as e:
            logger.error(f"TanML error: {e}")
    """
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}
    
    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


# ============================================================================
# Check Runner Exceptions
# ============================================================================

class CheckRunnerError(TanMLError):
    """Base exception for check runner errors."""
    
    def __init__(
        self,
        message: str,
        check_name: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, details)
        self.check_name = check_name


class CheckNotFoundError(CheckRunnerError):
    """Raised when a requested check is not found in the registry."""
    
    def __init__(self, check_name: str):
        super().__init__(
            f"Check '{check_name}' not found in registry",
            check_name=check_name,
        )


class CheckExecutionError(CheckRunnerError):
    """Raised when a check fails during execution."""
    
    def __init__(
        self,
        check_name: str,
        original_error: Exception,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            f"Check '{check_name}' failed: {original_error}",
            check_name=check_name,
            details=details,
        )
        self.original_error = original_error


class CheckConfigurationError(CheckRunnerError):
    """Raised when check configuration is invalid."""
    
    def __init__(
        self,
        check_name: str,
        config_key: str,
        message: str,
    ):
        super().__init__(
            f"Invalid configuration for '{check_name}': {message}",
            check_name=check_name,
            details={"config_key": config_key},
        )


# ============================================================================
# Report Generation Exceptions
# ============================================================================

class ReportGenerationError(TanMLError):
    """Base exception for report generation errors."""
    pass


class TemplateNotFoundError(ReportGenerationError):
    """Raised when the report template file is not found."""
    
    def __init__(self, template_path: str):
        super().__init__(
            f"Report template not found: {template_path}",
            details={"template_path": template_path},
        )


class SectionBuildError(ReportGenerationError):
    """Raised when a report section fails to build."""
    
    def __init__(
        self,
        section_name: str,
        original_error: Exception,
    ):
        super().__init__(
            f"Failed to build report section '{section_name}': {original_error}",
            details={"section_name": section_name},
        )
        self.section_name = section_name
        self.original_error = original_error


# ============================================================================
# Configuration Exceptions
# ============================================================================

class ConfigurationError(TanMLError):
    """Base exception for configuration errors."""
    pass


class ValidationError(ConfigurationError):
    """Raised when configuration validation fails."""
    
    def __init__(self, field: str, message: str, value: Any = None):
        super().__init__(
            f"Validation error for '{field}': {message}",
            details={"field": field, "value": value},
        )


# ============================================================================
# Data Loading Exceptions
# ============================================================================

class DataLoadError(TanMLError):
    """Raised when data loading fails."""
    
    def __init__(
        self,
        path: str,
        file_type: str,
        original_error: Optional[Exception] = None,
    ):
        message = f"Failed to load {file_type} file: {path}"
        if original_error:
            message += f" ({original_error})"
        super().__init__(
            message,
            details={"path": path, "file_type": file_type},
        )
        self.original_error = original_error
