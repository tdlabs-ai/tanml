# tanml/ui/pages/evaluation/__init__.py
"""
Model Evaluation Page - Modular Tab Architecture.

This package implements the evaluation page using a plugin-based tab system.
Contributors can add new tabs by creating files in the tabs/ folder.

Usage:
    from tanml.ui.views.evaluation import render_model_evaluation_page
    render_model_evaluation_page(run_dir)
"""

from tanml.ui.views.evaluation.main import render_model_evaluation_page

__all__ = ["render_model_evaluation_page"]
