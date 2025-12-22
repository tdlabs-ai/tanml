# tanml/ui/services/session.py
"""
Session state management for TanML UI.

Provides utilities for managing Streamlit session state with
persistent storage for validation runs.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import streamlit as st


def get_session_id() -> str:
    """
    Get or create a unique session ID.
    
    The session ID persists across Streamlit reruns but
    is unique per browser session.
    
    Returns:
        Unique session identifier string
    """
    if "session_id" not in st.session_state:
        # Create deterministic ID based on timestamp and random component
        import uuid
        st.session_state.session_id = str(uuid.uuid4())[:8]
    return st.session_state.session_id


def get_session_dir() -> Path:
    """
    Get the directory for storing session artifacts.
    
    Creates a dated directory structure for organizing validation runs:
    tmp/tanml_runs/YYYYMMDD/SESSION_ID/
    
    Returns:
        Path to the session directory
    """
    if "run_dir" not in st.session_state:
        base = Path(tempfile.gettempdir()) / "tanml_runs"
        date_str = datetime.now().strftime("%Y%m%d")
        session_id = get_session_id()
        
        run_dir = base / date_str / session_id
        run_dir.mkdir(parents=True, exist_ok=True)
        
        st.session_state.run_dir = run_dir
    
    return st.session_state.run_dir


def get_session_value(key: str, default: Any = None) -> Any:
    """
    Get a value from session state.
    
    Args:
        key: Session state key
        default: Default value if key not found
        
    Returns:
        The stored value or default
    """
    return st.session_state.get(key, default)


def set_session_value(key: str, value: Any) -> None:
    """
    Set a value in session state.
    
    Args:
        key: Session state key
        value: Value to store
    """
    st.session_state[key] = value


def clear_session(*keys: str) -> None:
    """
    Clear specific keys or entire session state.
    
    Args:
        *keys: Keys to clear. If empty, clears all session state.
    """
    if keys:
        for key in keys:
            if key in st.session_state:
                del st.session_state[key]
    else:
        st.session_state.clear()


def get_or_create(key: str, factory: callable) -> Any:
    """
    Get a session value, creating it if it doesn't exist.
    
    Args:
        key: Session state key
        factory: Callable that creates the default value
        
    Returns:
        The stored or newly created value
    """
    if key not in st.session_state:
        st.session_state[key] = factory()
    return st.session_state[key]


# Common session keys as constants
SESSION_KEYS = {
    "CLEANED_DF": "tvr_cleaned_df",
    "RAW_DF": "tvr_raw_df",
    "TARGET_COL": "tvr_target_col",
    "MODEL": "tvr_model",
    "RESULTS": "tvr_results",
    "TASK_TYPE": "tvr_task_type",
    "RUN_DIR": "run_dir",
}


def get_cleaned_df():
    """Get the cleaned DataFrame from session."""
    return get_session_value(SESSION_KEYS["CLEANED_DF"])


def set_cleaned_df(df):
    """Set the cleaned DataFrame in session."""
    set_session_value(SESSION_KEYS["CLEANED_DF"], df)


def get_model():
    """Get the trained model from session."""
    return get_session_value(SESSION_KEYS["MODEL"])


def set_model(model):
    """Set the trained model in session."""
    set_session_value(SESSION_KEYS["MODEL"], model)


def get_results():
    """Get the validation results from session."""
    return get_session_value(SESSION_KEYS["RESULTS"])


def set_results(results):
    """Set the validation results in session."""
    set_session_value(SESSION_KEYS["RESULTS"], results)
