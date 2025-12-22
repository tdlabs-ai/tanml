# tanml/config/__init__.py
"""
Configuration management for TanML.

This module provides centralized configuration handling with:
- Environment variable support (TANML_* prefix)
- Default values for all settings
- Type-safe configuration access
- Easy override for testing

Usage:
    from tanml.config import settings
    
    # Access settings
    artifacts_dir = settings.artifacts_dir
    max_upload_mb = settings.max_upload_mb
    
    # Override for testing
    from tanml.config import TanMLSettings
    test_settings = TanMLSettings(artifacts_dir="/tmp/test")
"""

from tanml.config.settings import TanMLSettings, settings

__all__ = [
    "TanMLSettings",
    "settings",
]
