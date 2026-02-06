# tanml/ui/__init__.py
"""
TanML Streamlit UI Package.

This package contains a modular, SOLID-compliant UI architecture:

Structure:
    components/     - Reusable UI widgets
        metrics.py  - KPI displays, metric cards
        forms.py    - Configuration forms, file uploaders
        renderers.py - Specialized result renderers

    pages/          - Page-level components
        setup.py    - Data upload page
        profiling.py - Data exploration page

    services/       - Business logic layer
        session.py  - Session state management
        data.py     - Data loading/validation
        validation.py - Validation orchestration
        cv.py       - Cross-validation logic
        tvr.py      - Train/Validate/Report workflow
        rule_config.py - Configuration building

    config.py       - Styling and theming
    helpers.py      - Utility functions
    narratives.py   - Report narrative generation
    glossary.py     - Technical term definitions
    reports.py      - Document generation

For Contributors:
    - Add new widgets to components/
    - Add page-level logic to pages/
    - Add business logic to services/

Main Entry Point:
    streamlit run tanml/ui/app.py
"""

from tanml.ui.config import (
    APP_ICON,
    APP_TITLE,
    COLORS,
    configure_page,
    get_status_color,
    load_css,
)
from tanml.ui.glossary import GLOSSARY
from tanml.ui.helpers import (
    fmt2,
    get_value_or_default,
    pick_target,
    save_upload,
    session_dir,
)
from tanml.ui.narratives import (
    story_drift,
    story_features,
    story_overfitting,
    story_performance,
    story_shap,
    story_stress,
)

__all__ = [
    "APP_ICON",
    # Config
    "APP_TITLE",
    "COLORS",
    # Glossary
    "GLOSSARY",
    "configure_page",
    "fmt2",
    "get_status_color",
    "get_value_or_default",
    "load_css",
    "pick_target",
    "save_upload",
    # Helpers
    "session_dir",
    "story_drift",
    "story_features",
    "story_overfitting",
    # Narratives
    "story_performance",
    "story_shap",
    "story_stress",
]


def run_app():
    """
    Run the TanML Streamlit application.

    This is an alternative entry point that can be called programmatically.

    Example:
        from tanml.ui import run_app
        run_app()
    """
    import subprocess
    import sys
    from pathlib import Path

    app_path = Path(__file__).parent / "app.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)])
