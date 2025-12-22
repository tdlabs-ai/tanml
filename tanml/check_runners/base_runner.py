# tanml/check_runners/base_runner.py
"""
Base class for all check runners.

This module provides the foundational class that all check runners inherit from,
ensuring a consistent interface and common functionality across all checks.

For Contributors:
    To create a new check runner, inherit from BaseCheckRunner and implement:
    1. `name` property - unique identifier for your check
    2. `execute()` method - your check logic
    
    Example:
        from tanml.check_runners.base_runner import BaseCheckRunner
        
        class MyCustomCheckRunner(BaseCheckRunner):
            @property
            def name(self) -> str:
                return "MyCustomCheck"
            
            def execute(self) -> Dict[str, Any]:
                # Your check logic here
                df = self.context.cleaned_df
                result = {"status": "ok", "rows": len(df)}
                return result
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

if TYPE_CHECKING:
    from tanml.core.context import CheckContext


class BaseCheckRunner(ABC):
    """
    Abstract base class for all check runners.
    
    Provides common functionality including:
    - Configuration access via `self._config`
    - Output directory management via `get_output_dir()`
    - Standardized error handling in `run()`
    - Enabled/disabled checks via `enabled` property
    
    The Template Method pattern is used: subclasses implement `execute()`,
    and the base class handles setup, validation, and error handling in `run()`.
    
    Attributes:
        context: The CheckContext containing all data and configuration
        
    Properties:
        name: Unique identifier for this check (abstract, must implement)
        enabled: Whether this check is enabled based on configuration
        
    Methods:
        run(): Template method that calls execute() with error handling
        execute(): Abstract method - implement your check logic here
        get_output_dir(): Get/create output directory for artifacts
    """
    
    def __init__(self, context: "CheckContext"):
        """
        Initialize the check runner with context.
        
        Args:
            context: CheckContext containing model, data, and configuration
        """
        self.context = context
        self._config: Dict[str, Any] = context.config.get(self.name, {})
        # Also check for legacy config keys
        legacy_key = self.name.replace("Check", "").lower()
        if not self._config:
            self._config = context.config.get(legacy_key, {})
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Unique identifier for this check.
        
        This is used for:
        - Configuration lookup (config[name])
        - Result dictionary keys
        - Error messages and logging
        
        Returns:
            String identifier like "CorrelationCheck", "EDACheck", etc.
        """
        pass
    
    @property
    def enabled(self) -> bool:
        """
        Check if this runner is enabled in configuration.
        
        Looks for 'enabled' key in the check's config section.
        Defaults to True if not specified.
        
        Returns:
            True if the check should run, False otherwise
        """
        return bool(self._config.get("enabled", True))
    
    def get_output_dir(self, subdir: str = "") -> Path:
        """
        Get the output directory for this check's artifacts.
        
        Creates the directory if it doesn't exist.
        
        Args:
            subdir: Optional subdirectory within the check's output folder
            
        Returns:
            Path to the output directory
            
        Example:
            # Get base output dir for this check
            output = self.get_output_dir()  # e.g., reports/correlation
            
            # Get subdirectory
            plots = self.get_output_dir("plots")  # e.g., reports/correlation/plots
        """
        # Check for configured artifacts directory
        base_dir = (
            self.context.config.get("options", {}).get("save_artifacts_dir")
            or self.context.config.get("paths", {}).get("artifacts_dir")
            or str(self.context.artifacts_dir)
        )
        
        # Build path: base / check_name / subdir
        check_dir = self.name.lower().replace("check", "")
        path = Path(base_dir) / check_dir
        if subdir:
            path = path / subdir
        
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    def get_config_value(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value for this check.
        
        Args:
            key: Configuration key to look up
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        return self._config.get(key, default)
    
    @abstractmethod
    def execute(self) -> Dict[str, Any]:
        """
        Execute the check logic.
        
        Implement this method in your subclass with your check's core logic.
        You have access to:
        - self.context: The full CheckContext
        - self._config: This check's configuration section
        - self.get_output_dir(): For saving artifacts
        - self.get_config_value(): For reading configuration
        
        Returns:
            Dictionary containing the check results.
            Should include relevant metrics, file paths, and status.
            
        Raises:
            Any exception will be caught by run() and returned as an error dict.
        """
        pass
    
    def run(
        self,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Execute the check with standardized error handling.
        
        This is the template method that:
        1. Checks if the runner is enabled
        2. Calls the progress callback if provided
        3. Executes the check via execute()
        4. Catches and wraps any errors
        
        Args:
            progress_callback: Optional callback to report progress
            
        Returns:
            Check results dictionary, or None if disabled.
            On error, returns {self.name: {"error": str(e)}}.
        """
        if not self.enabled:
            print(f"ℹ️ {self.name} skipped (disabled)")
            return None
        
        if progress_callback:
            progress_callback(f"Running {self.name}...")
        
        try:
            return self.execute()
        except Exception as e:
            error_msg = f"⚠️ {self.name} failed: {e}"
            print(error_msg)
            return {self.name: {"error": str(e)}}
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name={self.name}, enabled={self.enabled})>"


# =============================================================================
# Legacy Compatibility
# =============================================================================

def create_runner_from_function(
    name: str,
    func: Callable[..., Dict[str, Any]],
) -> type:
    """
    Create a BaseCheckRunner subclass from a legacy function.
    
    This helper allows gradual migration from function-based runners
    to class-based runners.
    
    Args:
        name: Check name
        func: Legacy runner function
        
    Returns:
        A new class that wraps the function
        
    Example:
        from tanml.check_runners.legacy_runner import run_my_check
        MyCheckRunner = create_runner_from_function("MyCheck", run_my_check)
    """
    
    class LegacyRunner(BaseCheckRunner):
        @property
        def name(self) -> str:
            return name
        
        def execute(self) -> Dict[str, Any]:
            return func(
                model=self.context.model,
                X_train=self.context.X_train,
                X_test=self.context.X_test,
                y_train=self.context.y_train,
                y_test=self.context.y_test,
                config=self.context.config,
                cleaned_df=self.context.cleaned_df,
                raw_df=self.context.raw_df,
            )
    
    LegacyRunner.__name__ = f"{name}Runner"
    LegacyRunner.__qualname__ = f"{name}Runner"
    return LegacyRunner
