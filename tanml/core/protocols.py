# tanml/core/protocols.py
"""
Protocol definitions for TanML components.

Protocols enable structural subtyping (duck typing) with type checker support.
Contributors can implement these interfaces without inheriting from base classes,
making it easy to integrate custom components.

Example:
    # A custom check that implements CheckRunnerProtocol
    class MyCheck:
        def __init__(self, context):
            self._context = context
        
        @property
        def name(self) -> str:
            return "MyCheck"
        
        @property
        def enabled(self) -> bool:
            return True
        
        def run(self) -> Dict[str, Any]:
            return {"result": "ok"}
    
    # This will type-check correctly even without inheritance
    check: CheckRunnerProtocol = MyCheck(context)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional, Protocol, runtime_checkable

if TYPE_CHECKING:
    from pathlib import Path
    from tanml.core.context import CheckContext, ReportContext


@runtime_checkable
class CheckRunnerProtocol(Protocol):
    """
    Protocol for validation check runners.
    
    All check runners must implement this interface to be compatible with
    the TanML validation engine. The `@runtime_checkable` decorator allows
    isinstance() checks at runtime.
    
    Attributes:
        name: Unique identifier for this check (e.g., "CorrelationCheck")
        enabled: Whether this check should run based on configuration
    
    Methods:
        run: Execute the check and return results dictionary
    """
    
    @property
    def name(self) -> str:
        """Unique identifier for this check."""
        ...
    
    @property
    def enabled(self) -> bool:
        """Whether this check is enabled in configuration."""
        ...
    
    def run(self) -> Optional[Dict[str, Any]]:
        """
        Execute the check and return results.
        
        Returns:
            Dictionary containing check results, or None if skipped/disabled.
            The dictionary should include at minimum a status indicator.
            On error, return {"error": "description"}.
        """
        ...


@runtime_checkable
class ReportSectionBuilderProtocol(Protocol):
    """
    Protocol for report section builders.
    
    Each section of the validation report (metadata, performance, EDA, etc.)
    is built by a component implementing this protocol.
    
    Attributes:
        section_name: Identifier for this section (e.g., "performance")
    
    Methods:
        should_include: Determine if this section should be included
        build: Build the section content into the document
    """
    
    @property
    def section_name(self) -> str:
        """Identifier for this report section."""
        ...
    
    def should_include(self, context: "ReportContext") -> bool:
        """
        Determine if this section should be included in the report.
        
        Args:
            context: Report generation context with results data
            
        Returns:
            True if the section should be included, False otherwise
        """
        ...
    
    def build(self, context: "ReportContext", document: Any) -> None:
        """
        Build this section into the document.
        
        Args:
            context: Report generation context with results data
            document: The document object to modify (e.g., python-docx Document)
        """
        ...


@runtime_checkable
class DataLoaderProtocol(Protocol):
    """
    Protocol for data loaders supporting various file formats.
    
    Contributors can implement this to add support for new file formats.
    """
    
    @property
    def supported_extensions(self) -> tuple[str, ...]:
        """File extensions this loader supports (e.g., ('.csv', '.tsv'))."""
        ...
    
    def can_load(self, path: "Path") -> bool:
        """Check if this loader can handle the given file."""
        ...
    
    def load(self, path: "Path", **kwargs) -> Any:
        """
        Load data from the file.
        
        Args:
            path: Path to the file
            **kwargs: Format-specific options
            
        Returns:
            Loaded data (typically a pandas DataFrame)
        """
        ...


@runtime_checkable
class FormatterProtocol(Protocol):
    """
    Protocol for value formatters used in reports.
    
    Formatters convert raw values to display-ready strings.
    """
    
    def format(self, value: Any) -> str:
        """
        Format a value for display.
        
        Args:
            value: Raw value to format
            
        Returns:
            Formatted string representation
        """
        ...
