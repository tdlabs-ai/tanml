# tanml/ui/reports/__init__.py
"""
Modular report generation for TanML.

This module provides a plugin-based architecture for generating Word reports.
Contributors can add new report sections by implementing the ReportSection protocol.

Example:
    from tanml.ui.reports import generate_dev_report, generate_eval_report
    
    # Generate development report
    doc_bytes = generate_dev_report(model_data, cv_results)
    
    # Generate evaluation report  
    doc_bytes = generate_eval_report(eval_data)
"""

from tanml.ui.reports.base import (
    ReportContext,
    ReportSection,
    SectionRegistry,
    add_table_with_borders,
    add_image_from_figure,
)

__all__ = [
    "ReportContext",
    "ReportSection",
    "SectionRegistry",
    "add_table_with_borders",
    "add_image_from_figure",
]
