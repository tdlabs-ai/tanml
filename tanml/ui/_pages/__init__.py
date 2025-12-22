# tanml/ui/pages/__init__.py
"""
Page modules for TanML UI.

Each page module provides a `render()` function that displays
that page's content when called.

Pages:
    - setup: Data upload and initial configuration
    - data_profiling: EDA and data exploration
    - preprocessing: Data cleaning and transformation
    - feature_ranking: Feature importance and ranking
    - model_development: Model training and tuning
    - model_evaluation: Validation results and reports

For Contributors:
    When adding a new page:
    1. Create a new file in this directory (e.g., my_page.py)
    2. Implement a `render(run_dir: Path)` function
    3. Import and register it in app.py
"""

__all__ = []
