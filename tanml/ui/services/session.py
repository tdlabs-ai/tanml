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
    d = Path("tanml_runs") / sid
    d.mkdir(parents=True, exist_ok=True)
    (d / "artifacts").mkdir(parents=True, exist_ok=True)
    return d

def _update_report_buffer(phase, data):
    """Update the global report buffer in session state."""
    if "report_buffer" not in st.session_state:
        st.session_state["report_buffer"] = {}
    if phase not in st.session_state["report_buffer"]:
        st.session_state["report_buffer"][phase] = {}
    
    # Merge if possible, else assign
    if isinstance(st.session_state["report_buffer"][phase], dict) and isinstance(data, dict):
         st.session_state["report_buffer"][phase].update(data)
    else:
         st.session_state["report_buffer"][phase] = data
