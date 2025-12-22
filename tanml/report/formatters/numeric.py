# tanml/report/formatters/numeric.py
"""
Numeric formatting utilities for reports.

Provides consistent number formatting across all reports.
"""

from __future__ import annotations

from typing import Any, Optional


def fmt_number(
    x: Any,
    decimals: int = 2,
    dash: str = "—",
) -> str:
    """
    Format a numeric value with specified decimal places.
    
    Args:
        x: Value to format (can be None, NaN, or any numeric type)
        decimals: Number of decimal places
        dash: String to return for None/NaN values
        
    Returns:
        Formatted string
        
    Examples:
        >>> fmt_number(0.12345, decimals=2)
        '0.12'
        >>> fmt_number(None)
        '—'
        >>> fmt_number(float('nan'))
        '—'
    """
    try:
        if x is None:
            return dash
        xf = float(x)
        if xf != xf:  # NaN check
            return dash
        return f"{xf:.{decimals}f}"
    except Exception:
        return dash


def fmt_ratio_or_pct(
    x: Any,
    pct: bool = False,
    decimals: int = 2,
    dash: str = "—",
) -> str:
    """
    Format a ratio, optionally as percentage.
    
    Args:
        x: Value to format (expected to be between 0 and 1 for ratios)
        pct: If True, multiply by 100 and add % suffix
        decimals: Number of decimal places
        dash: String to return for None/NaN values
        
    Returns:
        Formatted string
        
    Examples:
        >>> fmt_ratio_or_pct(0.25, pct=True)
        '25.00%'
        >>> fmt_ratio_or_pct(0.25, pct=False)
        '0.25'
    """
    try:
        if x is None:
            return dash
        xf = float(x)
        if xf != xf:  # NaN
            return dash
        if pct:
            return f"{xf * 100:.{decimals}f}%"
        return f"{xf:.{decimals}f}"
    except Exception:
        return dash


def fmt_int(x: Any, dash: str = "—") -> str:
    """
    Format a value as an integer.
    
    Args:
        x: Value to format
        dash: String to return for None/NaN values
        
    Returns:
        Formatted integer string
        
    Examples:
        >>> fmt_int(42.7)
        '43'
        >>> fmt_int(None)
        '—'
    """
    try:
        if x is None:
            return dash
        return str(int(round(float(x))))
    except Exception:
        return dash


def fmt_float(x: Any, decimals: int = 2, dash: str = "—") -> str:
    """
    Format a value as a float with fixed decimal places.
    
    Alias for fmt_number with clearer naming.
    
    Args:
        x: Value to format
        decimals: Number of decimal places
        dash: String to return for None/NaN values
        
    Returns:
        Formatted float string
    """
    return fmt_number(x, decimals=decimals, dash=dash)


# Legacy compatibility aliases
_fmt2 = fmt_number
_fmt_ratio_or_pct = fmt_ratio_or_pct
