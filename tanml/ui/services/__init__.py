# tanml/ui/services/__init__.py
"""
Business logic services for TanML UI.

These services handle the core business logic that the UI calls,
keeping the UI layer thin and focused on presentation.

Services:
    - session: Session state management
    - data: Data loading and preprocessing
    - validation: Validation orchestration
    - report: Report generation
"""

from tanml.ui.services.session import (
    get_session_dir,
    get_session_value,
    set_session_value,
    clear_session,
)
from tanml.ui.services.data import (
    save_uploaded_file,
    load_data_file,
)

__all__ = [
    # Session management
    "get_session_dir",
    "get_session_value",
    "set_session_value",
    "clear_session",
    # Data handling
    "save_uploaded_file",
    "load_data_file",
]
