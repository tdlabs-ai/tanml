# tanml/utils/plotting/save.py
"""
Figure saving utilities for TanML.

Provides consistent figure saving with proper paths and formats.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt


def get_figure_path(
    output_dir: Union[str, Path],
    filename: str,
    extension: str = "png",
) -> Path:
    """
    Get the full path for saving a figure.
    
    Creates the output directory if it doesn't exist.
    
    Args:
        output_dir: Directory to save to
        filename: Base filename (without extension)
        extension: File extension (default: png)
        
    Returns:
        Full Path object for the file
        
    Example:
        path = get_figure_path("reports/eda", "histogram")
        # Returns: Path("reports/eda/histogram.png")
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Ensure filename doesn't have extension already
    if filename.endswith(f".{extension}"):
        return output_dir / filename
    
    return output_dir / f"{filename}.{extension}"


def save_figure(
    fig: plt.Figure,
    output_dir: Union[str, Path],
    filename: str,
    dpi: int = 200,
    format: str = "png",
    close: bool = True,
    **kwargs,
) -> str:
    """
    Save a matplotlib figure to disk.
    
    Handles directory creation, proper DPI settings, and cleanup.
    
    Args:
        fig: matplotlib Figure to save
        output_dir: Directory to save to
        filename: Base filename (without extension)
        dpi: Resolution in dots per inch
        format: Image format (png, pdf, svg, etc.)
        close: Whether to close the figure after saving
        **kwargs: Additional arguments for fig.savefig()
        
    Returns:
        String path to the saved file
        
    Example:
        from tanml.utils.plotting import save_figure
        
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])
        path = save_figure(fig, "reports/plots", "quadratic")
    """
    path = get_figure_path(output_dir, filename, extension=format)
    
    # Default save settings
    save_kwargs = {
        "dpi": dpi,
        "bbox_inches": "tight",
        "pad_inches": 0.1,
        "facecolor": "white",
        "edgecolor": "none",
    }
    save_kwargs.update(kwargs)
    
    fig.savefig(path, **save_kwargs)
    
    if close:
        plt.close(fig)
    
    return str(path)


def close_all_figures() -> None:
    """
    Close all open matplotlib figures.
    
    Useful for cleanup after running multiple checks.
    """
    plt.close("all")
