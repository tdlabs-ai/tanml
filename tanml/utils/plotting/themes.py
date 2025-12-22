# tanml/utils/plotting/themes.py
"""
TanML plotting themes and colors.

Provides consistent styling across all visualizations in TanML.
"""

from __future__ import annotations

from typing import List, Optional

import matplotlib.pyplot as plt


# TanML brand colors
TANML_COLORS = {
    "primary": "#2563eb",      # Blue
    "secondary": "#7c3aed",    # Purple
    "success": "#10b981",      # Green
    "warning": "#f59e0b",      # Amber
    "danger": "#ef4444",       # Red
    "info": "#06b6d4",         # Cyan
    "neutral": "#6b7280",      # Gray
    "background": "#ffffff",   # White
    "text": "#1f2937",         # Dark gray
}

# Color palettes for multi-series plots
TANML_PALETTE = [
    "#2563eb",  # Blue
    "#7c3aed",  # Purple
    "#10b981",  # Green
    "#f59e0b",  # Amber
    "#ef4444",  # Red
    "#06b6d4",  # Cyan
    "#f97316",  # Orange
    "#8b5cf6",  # Violet
    "#14b8a6",  # Teal
    "#ec4899",  # Pink
]


def set_tanml_style(
    style: str = "seaborn-v0_8-whitegrid",
    font_scale: float = 1.0,
) -> None:
    """
    Apply TanML's consistent plot styling.
    
    Call this at the start of any check that creates plots to ensure
    consistent appearance across all visualizations.
    
    Args:
        style: matplotlib style to use as base
        font_scale: Scale factor for font sizes
        
    Example:
        from tanml.utils.plotting import set_tanml_style
        
        set_tanml_style()
        fig, ax = plt.subplots()
        ax.plot(x, y)
    """
    # Try to use the style, fall back to a basic one if not available
    available_styles = plt.style.available
    if style in available_styles:
        plt.style.use(style)
    elif "seaborn" in available_styles:
        plt.style.use("seaborn")
    else:
        plt.style.use("ggplot")
    
    # Apply TanML customizations
    plt.rcParams.update({
        # Colors
        "axes.prop_cycle": plt.cycler(color=TANML_PALETTE),
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "axes.edgecolor": "#e5e7eb",
        "axes.labelcolor": TANML_COLORS["text"],
        "xtick.color": TANML_COLORS["text"],
        "ytick.color": TANML_COLORS["text"],
        "text.color": TANML_COLORS["text"],
        
        # Fonts
        "font.size": 10 * font_scale,
        "axes.titlesize": 12 * font_scale,
        "axes.labelsize": 10 * font_scale,
        "xtick.labelsize": 9 * font_scale,
        "ytick.labelsize": 9 * font_scale,
        "legend.fontsize": 9 * font_scale,
        
        # Figure
        "figure.dpi": 100,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
        
        # Grid
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
    })
    
    # Try to set seaborn palette if available
    try:
        import seaborn as sns
        sns.set_palette(TANML_PALETTE)
    except ImportError:
        pass


def get_color(name: str) -> str:
    """
    Get a TanML color by name.
    
    Args:
        name: Color name (e.g., "primary", "success", "danger")
        
    Returns:
        Hex color code
        
    Raises:
        KeyError: If color name not found
    """
    return TANML_COLORS[name]


def get_palette(n_colors: int = 10) -> List[str]:
    """
    Get a color palette with n colors.
    
    Args:
        n_colors: Number of colors needed
        
    Returns:
        List of hex color codes
    """
    if n_colors <= len(TANML_PALETTE):
        return TANML_PALETTE[:n_colors]
    
    # Repeat palette if more colors needed
    extended = TANML_PALETTE * (n_colors // len(TANML_PALETTE) + 1)
    return extended[:n_colors]
