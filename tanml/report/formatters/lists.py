# tanml/report/formatters/lists.py
"""
List and collection formatting utilities for reports.

Provides consistent formatting for lists, feature names, and other collections.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union


def fmt_list(
    lst: Any,
    max_items: int = 20,
    sep: str = ", ",
) -> str:
    """
    Format a list as a string with optional truncation.
    
    Args:
        lst: List or iterable to format
        max_items: Maximum items to show before truncating
        sep: Separator between items
        
    Returns:
        Formatted string
        
    Examples:
        >>> fmt_list(['a', 'b', 'c'])
        'a, b, c'
        >>> fmt_list(['a', 'b', 'c', 'd'], max_items=2)
        'a, b, … (+2 more)'
    """
    if not lst:
        return "—"
    if isinstance(lst, (str, bytes)):
        return str(lst)
    
    try:
        seq = list(lst)
    except Exception:
        return str(lst)
    
    if len(seq) <= max_items:
        return sep.join(map(str, seq))
    
    head = sep.join(map(str, seq[:max_items]))
    return f"{head}{sep}… (+{len(seq) - max_items} more)"


def fmt_list_or_message(
    lst: Any,
    empty_msg: str = "None (no issues detected)",
    max_items: int = 20,
    sep: str = ", ",
) -> str:
    """
    Format a list, or return a message if empty.
    
    Args:
        lst: List or iterable to format
        empty_msg: Message to return if list is empty
        max_items: Maximum items to show before truncating
        sep: Separator between items
        
    Returns:
        Formatted string or empty message
        
    Examples:
        >>> fmt_list_or_message([])
        'None (no issues detected)'
        >>> fmt_list_or_message(['col1', 'col2'], empty_msg='All good!')
        'col1, col2'
    """
    if not lst:
        return empty_msg
    
    try:
        seq = list(lst) if not isinstance(lst, (str, bytes)) else [lst]
    except Exception:
        return str(lst)
    
    if len(seq) <= max_items:
        return sep.join(map(str, seq))
    
    head = sep.join(map(str, seq[:max_items]))
    return f"{head}{sep}… (+{len(seq) - max_items} more)"


def fmt_feature_names(
    v: Any,
    max_names: int = 30,
) -> str:
    """
    Format a list of feature names.
    
    Args:
        v: List of feature names or single name
        max_names: Maximum names to show before truncating
        
    Returns:
        Formatted string
        
    Examples:
        >>> fmt_feature_names(['age', 'income', 'score'])
        'age, income, score'
    """
    if v is None:
        return ""
    
    if isinstance(v, (list, tuple)):
        if len(v) <= max_names:
            return ", ".join(map(str, v))
        head = ", ".join(map(str, v[:max_names]))
        return f"{head}, … (+{len(v) - max_names} more)"
    
    return str(v)


def fmt_target_balance(tb: Any) -> str:
    """
    Format target balance dictionary.
    
    For classification: class counts or percentages
    For regression: statistics (min, max, mean, etc.)
    
    Args:
        tb: Target balance dictionary
        
    Returns:
        Formatted string
        
    Examples:
        >>> fmt_target_balance({0: 0.7, 1: 0.3})
        '0: 70.0%, 1: 30.0%'
        >>> fmt_target_balance({'Mean': 45.2, 'Std': 12.1})
        'Mean: 45.2, Std: 12.1'
    """
    if not isinstance(tb, dict) or not tb:
        return ""
    
    vals = list(tb.values())
    
    # Check if values look like proportions (0-1 range)
    are_probs = all(
        isinstance(x, (int, float)) and 0 <= float(x) <= 1 
        for x in vals
    )
    
    items = []
    for k in sorted(tb.keys(), key=str):
        v = tb[k]
        if are_probs:
            try:
                items.append(f"{k}: {float(v) * 100:.1f}%")
            except Exception:
                items.append(f"{k}: {v}")
        else:
            items.append(f"{k}: {v}")
    
    return ", ".join(items)


# Legacy compatibility aliases
_fmt_list = fmt_list
_fmt_list_or_message = fmt_list_or_message
_fmt_feature_names = fmt_feature_names
_fmt_target_balance = fmt_target_balance
