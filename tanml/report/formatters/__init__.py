# tanml/report/formatters/__init__.py
"""
Formatters for report generation.

This module provides formatting utilities for converting raw values
into display-ready strings for reports.
"""

from tanml.report.formatters.numeric import (
    fmt_number,
    fmt_ratio_or_pct,
    fmt_int,
    fmt_float,
)
from tanml.report.formatters.lists import (
    fmt_list,
    fmt_list_or_message,
    fmt_feature_names,
    fmt_target_balance,
)

__all__ = [
    # Numeric formatters
    "fmt_number",
    "fmt_ratio_or_pct",
    "fmt_int",
    "fmt_float",
    # List formatters
    "fmt_list",
    "fmt_list_or_message",
    "fmt_feature_names",
    "fmt_target_balance",
]
