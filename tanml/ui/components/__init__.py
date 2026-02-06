# tanml/ui/components/__init__.py
"""
Reusable UI components for TanML.

This module provides small, reusable Streamlit widgets that can be
composed to build pages.

Components:
    - metrics: KPI displays, metric cards
    - forms: Model configuration forms, file uploaders
    - charts: Common visualization wrappers
    - tables: DataFrame display helpers
"""

from tanml.ui.components.forms import (
    render_model_form,
)
from tanml.ui.components.metrics import (
    kpi_row,
    metric_card,
    metric_no_trunc,
)

__all__ = [
    "kpi_row",
    # Metrics
    "metric_card",
    "metric_no_trunc",
    # Forms
    "render_model_form",
]
