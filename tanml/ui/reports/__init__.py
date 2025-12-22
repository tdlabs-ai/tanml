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
    add_table_with_borders,
    add_image_from_figure,
)

# Re-export all report generator functions from generators.py for backward compatibility
from tanml.ui.reports.generators import (
    _choose_report_template,
    _filter_metrics_for_task,
    _generate_dev_report_docx,
    _generate_eval_report_docx,
    _generate_ranking_report_docx,
    _fmt2,
)

__all__ = [
    "ReportContext",
    "ReportSection",
    "SectionRegistry",
    "add_table_with_borders",
    "add_image_from_figure",
    "_choose_report_template",
    "_filter_metrics_for_task",
    "_generate_dev_report_docx",
    "_generate_eval_report_docx",
    "_generate_ranking_report_docx",
    "_fmt2",
]
