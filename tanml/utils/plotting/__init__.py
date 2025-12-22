# tanml/utils/plotting/__init__.py
"""
Plotting utilities for TanML.

This module provides common plotting functions used across checks,
ensuring consistent styling and output handling.
"""

from tanml.utils.plotting.themes import (
    set_tanml_style,
    TANML_COLORS,
    TANML_PALETTE,
)
from tanml.utils.plotting.save import (
    save_figure,
    get_figure_path,
)

__all__ = [
    # Theme
    "set_tanml_style",
    "TANML_COLORS",
    "TANML_PALETTE",
    # Saving
    "save_figure",
    "get_figure_path",
]
