# tanml/ui/services/tvr.py
"""
TVR (Train/Validate/Report) session management for TanML UI.

Provides state management for the validation workflow.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import streamlit as st


def tvr_key(section_id: str, name: str) -> str:
    """Generate a session state key for a TVR section."""
    return f"tvr_{section_id}_{name}"


def tvr_init(section_id: str) -> None:
    """
    Initialize TVR section state.
    
    Args:
        section_id: Unique section identifier
    """
    prefix = f"tvr_{section_id}_"
    
    defaults = {
        f"{prefix}history": [],
        f"{prefix}status": "idle",  # idle, running, ready, error
        f"{prefix}report_bytes": None,
        f"{prefix}report_path": None,
        f"{prefix}summary": {},
        f"{prefix}extras": {},
    }
    
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


def tvr_reset(section_id: str) -> None:
    """
    Reset TVR section to initial state.
    
    Args:
        section_id: Section to reset
    """
    prefix = f"tvr_{section_id}_"
    
    keys_to_reset = [
        f"{prefix}status",
        f"{prefix}report_bytes",
        f"{prefix}report_path",
        f"{prefix}summary",
        f"{prefix}extras",
    ]
    
    for key in keys_to_reset:
        if key in st.session_state:
            if "status" in key:
                st.session_state[key] = "idle"
            elif "summary" in key or "extras" in key:
                st.session_state[key] = {}
            else:
                st.session_state[key] = None


def tvr_clear_extras(section_id: str) -> None:
    """
    Clear only the extras for a section.
    
    Args:
        section_id: Section to clear
    """
    key = f"tvr_{section_id}_extras"
    st.session_state[key] = {}


def tvr_finish(
    section_id: str,
    *,
    report_path: Path = None,
    report_bytes: bytes = None,
    file_name: str,
    summary: dict = None,
    label: str = None,
    cfg: dict = None,
) -> None:
    """
    Mark a TVR section as finished with report data.
    
    Args:
        section_id: Section identifier
        report_path: Path to generated report file
        report_bytes: Report file contents as bytes
        file_name: Filename for download
        summary: Summary metrics dictionary
        label: Human-readable label
        cfg: Configuration used
    """
    prefix = f"tvr_{section_id}_"
    
    st.session_state[f"{prefix}status"] = "ready"
    st.session_state[f"{prefix}report_path"] = report_path
    st.session_state[f"{prefix}report_bytes"] = report_bytes
    st.session_state[f"{prefix}file_name"] = file_name
    st.session_state[f"{prefix}summary"] = summary or {}
    
    # Add to history
    from datetime import datetime
    history_key = f"{prefix}history"
    
    entry = {
        "timestamp": datetime.now().isoformat(),
        "file_name": file_name,
        "label": label,
        "summary": summary,
        "cfg": cfg,
    }
    
    if history_key not in st.session_state:
        st.session_state[history_key] = []
    
    st.session_state[history_key].append(entry)
    
    # Keep only last 10 entries
    st.session_state[history_key] = st.session_state[history_key][-10:]


def tvr_store_extras(section_id: str, extras: Dict[str, Any]) -> None:
    """
    Store extra data for a TVR section.
    
    Args:
        section_id: Section identifier
        extras: Dictionary of extra data to store
    """
    key = f"tvr_{section_id}_extras"
    if key not in st.session_state:
        st.session_state[key] = {}
    st.session_state[key].update(extras)


def tvr_get_extras(section_id: str) -> Dict[str, Any]:
    """
    Get extra data for a TVR section.
    
    Args:
        section_id: Section identifier
        
    Returns:
        Dictionary of extra data
    """
    key = f"tvr_{section_id}_extras"
    return st.session_state.get(key, {})


def tvr_render_ready(
    section_id: str,
    *,
    header_text: str = "Refit, Validate & Report",
) -> None:
    """
    Render the "ready" state with download button.
    
    Args:
        section_id: Section identifier
        header_text: Header text to display
    """
    prefix = f"tvr_{section_id}_"
    
    status = st.session_state.get(f"{prefix}status", "idle")
    
    if status != "ready":
        return
    
    report_bytes = st.session_state.get(f"{prefix}report_bytes")
    file_name = st.session_state.get(f"{prefix}file_name", "report.docx")
    summary = st.session_state.get(f"{prefix}summary", {})
    
    st.success(f"✅ {header_text} - Complete!")
    
    # Summary metrics
    if summary:
        cols = st.columns(min(len(summary), 4))
        for i, (key, value) in enumerate(list(summary.items())[:4]):
            with cols[i]:
                if isinstance(value, float):
                    st.metric(key.upper(), f"{value:.4f}")
                else:
                    st.metric(key.upper(), str(value))
    
    # Download button
    if report_bytes:
        st.download_button(
            "📥 Download Report",
            report_bytes,
            file_name=file_name,
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )


def tvr_render_history(
    section_id: str,
    *,
    title: str = "🗂️ Past Runs",
) -> None:
    """
    Render run history for a section.
    
    Args:
        section_id: Section identifier
        title: Expander title
    """
    prefix = f"tvr_{section_id}_"
    history = st.session_state.get(f"{prefix}history", [])
    
    if not history:
        return
    
    with st.expander(title, expanded=False):
        for i, entry in enumerate(reversed(history)):
            ts = entry.get("timestamp", "Unknown")
            label = entry.get("label", entry.get("file_name", f"Run {i+1}"))
            summary = entry.get("summary", {})
            
            st.write(f"**{label}** ({ts})")
            if summary:
                summary_str = ", ".join(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" 
                                        for k, v in list(summary.items())[:3])
                st.caption(summary_str)
            st.divider()


def tvr_render_extras(section_id: str) -> None:
    """
    Render extra data stored for a section.
    
    Args:
        section_id: Section identifier
    """
    extras = tvr_get_extras(section_id)
    
    if not extras:
        return
    
    for key, value in extras.items():
        if isinstance(value, dict):
            st.json(value)
        else:
            st.write(f"**{key}**: {value}")


# Legacy aliases
_tvr_key = tvr_key
tvr_clear_extras = tvr_clear_extras
tvr_reset = tvr_reset
tvr_init = tvr_init
tvr_finish = tvr_finish
tvr_render_ready = tvr_render_ready
tvr_render_history = tvr_render_history
tvr_store_extras = tvr_store_extras
tvr_render_extras = tvr_render_extras
