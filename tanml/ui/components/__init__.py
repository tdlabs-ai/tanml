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

from tanml.ui.components.metrics import (
    metric_card,
    kpi_row,
    metric_no_trunc,
)
from tanml.ui.components.forms import (
    render_model_form,
)

__all__ = [
    # Metrics
    "metric_card",
    "kpi_row",
    "metric_no_trunc",
    # Forms
    "render_model_form",
]
