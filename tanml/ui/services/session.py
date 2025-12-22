# tanml/ui/services/session.py
"""
Session management service for TanML UI.
"""

from __future__ import annotations

import uuid
from pathlib import Path
import streamlit as st


def _session_dir() -> Path:
    """Per-user ephemeral run directory with artifacts subfolder."""
    sid = st.session_state.get("_session_id")
    if not sid:
        sid = str(uuid.uuid4())[:8]
        st.session_state["_session_id"] = sid
    d = Path(".ui_runs") / sid
    d.mkdir(parents=True, exist_ok=True)
    (d / "artifacts").mkdir(parents=True, exist_ok=True)
    return d
