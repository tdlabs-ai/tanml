# tanml/core/context.py
"""
Context objects for TanML components.

Context objects provide a clean, immutable interface for passing data
between components. This reduces parameter coupling and makes it easy
to extend functionality without changing method signatures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


@dataclass(frozen=True)
class CheckContext:
    """
    Immutable context passed to all check runners.
    
    This object encapsulates all the data a check might need, providing
    a consistent interface regardless of which check is being run.
    
    Attributes:
        model: The trained model being validated
        X_train: Training features (DataFrame)
        X_test: Test features (DataFrame)
        y_train: Training target (Series)
        y_test: Test target (Series)
        config: Full configuration dictionary
        cleaned_df: Cleaned/processed DataFrame
        raw_df: Original raw DataFrame (optional)
        task_type: "classification" or "regression"
        artifacts_dir: Directory for saving artifacts (plots, CSVs, etc.)
        target_column: Name of the target column
        feature_columns: List of feature column names
    
    Example:
        context = CheckContext(
            model=my_model,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            config=config,
            cleaned_df=df,
            task_type="classification",
        )
        
        runner = CorrelationCheckRunner(context)
        results = runner.run()
    """
    
    model: Any
    X_train: "pd.DataFrame"
    X_test: "pd.DataFrame"
    y_train: "pd.Series"
    y_test: "pd.Series"
    config: Dict[str, Any]
    cleaned_df: "pd.DataFrame"
    raw_df: Optional["pd.DataFrame"] = None
    task_type: str = "classification"
    artifacts_dir: Path = field(default_factory=lambda: Path("reports"))
    target_column: Optional[str] = None
    feature_columns: List[str] = field(default_factory=list)
    
    def get_check_config(self, check_name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific check.
        
        Args:
            check_name: Name of the check (e.g., "CorrelationCheck")
            
        Returns:
            Configuration dictionary for the check, or empty dict if not found
        """
        return self.config.get(check_name, {})
    
    def get_option(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration option.
        
        Args:
            key: Dot-separated path (e.g., "options.save_artifacts_dir")
            default: Default value if not found
            
        Returns:
            The configuration value or default
        """
        parts = key.split(".")
        current = self.config
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default
        return current
    
    def with_artifacts_dir(self, path: Path) -> "CheckContext":
        """
        Create a new context with a different artifacts directory.
        
        Since CheckContext is frozen (immutable), this creates a copy
        with the modified value.
        """
        return CheckContext(
            model=self.model,
            X_train=self.X_train,
            X_test=self.X_test,
            y_train=self.y_train,
            y_test=self.y_test,
            config=self.config,
            cleaned_df=self.cleaned_df,
            raw_df=self.raw_df,
            task_type=self.task_type,
            artifacts_dir=path,
            target_column=self.target_column,
            feature_columns=self.feature_columns,
        )


@dataclass(frozen=True)
class ReportContext:
    """
    Immutable context for report generation.
    
    Contains all validation results and metadata needed to build
    the final report document.
    
    Attributes:
        results: Full validation results dictionary
        task_type: "classification" or "regression"
        model_name: Display name for the model
        generated_at: Timestamp string for report generation
        template_path: Path to the report template
        output_path: Path where the report will be saved
    """
    
    results: Dict[str, Any]
    task_type: str = "classification"
    model_name: str = "Model"
    generated_at: str = ""
    template_path: Optional[Path] = None
    output_path: Optional[Path] = None
    
    def get_section(self, section_name: str) -> Dict[str, Any]:
        """
        Get results for a specific section.
        
        Args:
            section_name: Name of the section (e.g., "CorrelationCheck")
            
        Returns:
            Section results dictionary, or empty dict if not found
        """
        return self.results.get(section_name, {})
    
    def has_section(self, section_name: str) -> bool:
        """Check if results contain data for a section."""
        section = self.results.get(section_name)
        return section is not None and section != {} and not section.get("error")
    
    @property
    def is_classification(self) -> bool:
        """Check if this is a classification task."""
        return "class" in self.task_type.lower()
    
    @property
    def is_regression(self) -> bool:
        """Check if this is a regression task."""
        return not self.is_classification
