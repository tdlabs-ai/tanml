# tanml/config/settings.py
"""
TanML Settings Module.

Provides application-wide settings with environment variable support.
All settings can be overridden via environment variables with TANML_ prefix.

Environment Variables:
    TANML_ARTIFACTS_DIR: Directory for saving artifacts (default: "reports")
    TANML_MAX_UPLOAD_MB: Maximum upload size in MB (default: 1024)
    TANML_SAMPLE_ROWS_DEFAULT: Default rows for sampling (default: 150000)
    TANML_CAST9: Enable CAST9 mode (default: false)
    TANML_DEBUG: Enable debug mode (default: false)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


def _get_env_bool(key: str, default: bool = False) -> bool:
    """Get boolean from environment variable."""
    val = os.environ.get(key, "").lower()
    if val in ("1", "true", "yes", "on"):
        return True
    if val in ("0", "false", "no", "off"):
        return False
    return default


def _get_env_int(key: str, default: int) -> int:
    """Get integer from environment variable."""
    val = os.environ.get(key)
    if val is not None:
        try:
            return int(val)
        except ValueError:
            pass
    return default


def _get_env_path(key: str, default: Path) -> Path:
    """Get Path from environment variable."""
    val = os.environ.get(key)
    if val is not None:
        return Path(val)
    return default


@dataclass
class TanMLSettings:
    """
    Application-wide settings for TanML.
    
    All settings can be overridden via environment variables with TANML_ prefix,
    or by passing values directly to the constructor.
    
    Attributes:
        artifacts_dir: Directory for saving artifacts (plots, CSVs, reports)
        templates_dir: Directory containing report templates
        max_upload_mb: Maximum file upload size in MB
        sample_rows_default: Default number of rows for sampling large datasets
        cast9_enabled: Enable CAST9 compatibility mode
        debug: Enable debug mode with verbose logging
        
    Example:
        # Use default settings
        from tanml.config import settings
        print(settings.artifacts_dir)
        
        # Override via environment
        os.environ["TANML_DEBUG"] = "true"
        
        # Override programmatically
        custom = TanMLSettings(debug=True, artifacts_dir=Path("/tmp"))
    """
    
    # Paths
    artifacts_dir: Path = field(
        default_factory=lambda: _get_env_path("TANML_ARTIFACTS_DIR", Path("reports"))
    )
    templates_dir: Path = field(
        default_factory=lambda: Path(__file__).parent.parent / "report" / "templates"
    )
    
    # Performance / Limits
    max_upload_mb: int = field(
        default_factory=lambda: _get_env_int("TANML_MAX_UPLOAD_MB", 1024)
    )
    sample_rows_default: int = field(
        default_factory=lambda: _get_env_int("TANML_SAMPLE_ROWS_DEFAULT", 150_000)
    )
    
    # Feature flags
    cast9_enabled: bool = field(
        default_factory=lambda: _get_env_bool("TANML_CAST9", False)
    )
    debug: bool = field(
        default_factory=lambda: _get_env_bool("TANML_DEBUG", False)
    )
    
    # Check defaults
    correlation_threshold: float = 0.80
    vif_threshold: float = 5.0
    shap_background_size: int = 100
    shap_test_size: int = 200
    
    # Report settings
    report_dpi: int = 200
    report_image_width_inches: float = 6.5
    
    def get_template_path(self, task_type: str) -> Path:
        """
        Get the appropriate report template for the task type.
        
        Args:
            task_type: "classification" or "regression"
            
        Returns:
            Path to the template file
        """
        template_name = (
            "report_template_reg.docx"
            if task_type == "regression"
            else "report_template_cls.docx"
        )
        return self.templates_dir / template_name
    
    def get_check_defaults(self, check_name: str) -> Dict[str, Any]:
        """
        Get default configuration for a specific check.
        
        Args:
            check_name: Name of the check (e.g., "CorrelationCheck")
            
        Returns:
            Dictionary of default values for the check
        """
        defaults = {
            "CorrelationCheck": {
                "method": "pearson",
                "high_corr_threshold": self.correlation_threshold,
                "heatmap_max_features_default": 20,
                "heatmap_max_features_limit": 60,
                "subset_strategy": "cluster",
                "sample_rows": self.sample_rows_default,
                "save_csv": True,
                "save_fig": True,
            },
            "VIFCheck": {
                "threshold": self.vif_threshold,
            },
            "SHAPCheck": {
                "background_size": self.shap_background_size,
                "test_size": self.shap_test_size,
            },
            "EDACheck": {
                "max_plots": -1,  # -1 means unlimited
            },
            "StressTestCheck": {
                "noise_fraction": 0.2,
            },
        }
        return defaults.get(check_name, {})
    
    def as_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary."""
        return {
            "artifacts_dir": str(self.artifacts_dir),
            "templates_dir": str(self.templates_dir),
            "max_upload_mb": self.max_upload_mb,
            "sample_rows_default": self.sample_rows_default,
            "cast9_enabled": self.cast9_enabled,
            "debug": self.debug,
            "correlation_threshold": self.correlation_threshold,
            "vif_threshold": self.vif_threshold,
        }


# Global settings instance (singleton pattern)
settings = TanMLSettings()
