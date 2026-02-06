# tanml/ui/reports/__init__.py
"""
Modular report generation for TanML.

This module provides a plugin-based architecture for generating Word reports.
Contributors can add new report sections by implementing the ReportSection protocol.
"""

from tanml.ui.reports.base import (
    ReportContext,
    ReportSection,
    SectionRegistry,
    add_image_from_figure,
    add_table_with_borders,
)

# Re-export all report generator functions from generators.py for backward compatibility
from tanml.ui.reports.generators import (
    _choose_report_template,
    _filter_metrics_for_task,
    _fmt2,
    _generate_dev_report_docx,
    _generate_eval_report_docx,
    _generate_ranking_report_docx,
)

__all__ = [
    "ReportContext",
    "ReportSection",
    "SectionRegistry",
    "_choose_report_template",
    "_filter_metrics_for_task",
    "_fmt2",
    "_generate_dev_report_docx",
    "_generate_eval_report_docx",
    "_generate_ranking_report_docx",
    "add_image_from_figure",
    "add_table_with_borders",
]
