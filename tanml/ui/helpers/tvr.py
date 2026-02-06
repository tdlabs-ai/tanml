# tanml/ui/helpers/tvr.py
"""
TVR (Train/Validate/Report) helper functions for TanML UI.

These functions manage the state machine for training, validating, and
generating reports in the Streamlit session state.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import streamlit as st


def _tvr_key(section_id: str, name: str) -> str:
    """Generate a unique session state key for TVR data."""
    return f"tvr::{section_id}::{name}"


def tvr_clear_extras(section_id: str) -> None:
    """Clear extras data for a section."""
    st.session_state.pop(_tvr_key(section_id, "extras"), None)


def tvr_init(section_id: str) -> None:
    """Initialize TVR state for a section with default values."""
    for k, v in {
        "stage": "idle",
        "bytes": None,
        "file": None,
        "ts": None,
        "summary": None,
        "label": None,
        "cfg": None,
    }.items():
        st.session_state.setdefault(_tvr_key(section_id, k), v)


def tvr_reset(section_id: str, filter_metrics_fn=None) -> None:
    """Start fresh but keep history. Also hard-reset all UI widgets to defaults."""
    tvr_init(section_id)

    # Reset TVR state (keep history)
    st.session_state[_tvr_key(section_id, "stage")] = "idle"
    st.session_state[_tvr_key(section_id, "bytes")] = None
    st.session_state[_tvr_key(section_id, "file")] = None
    st.session_state[_tvr_key(section_id, "ts")] = None
    st.session_state[_tvr_key(section_id, "summary")] = None
    st.session_state[_tvr_key(section_id, "label")] = None
    st.session_state[_tvr_key(section_id, "cfg")] = None
    st.session_state.pop(_tvr_key(section_id, "extras"), None)

    # Hard-reset: clear widget states so UI returns to coded defaults
    keys_to_clear = [
        # Uploaders
        "upl_cleaned",
        "upl_train",
        "upl_test",
        # Sidebar options
        "opt_eda",
        "opt_eda_max",
        "opt_corr",
        "opt_vif",
        "opt_modelmeta",
        "opt_stress",
        "opt_stress_eps",
        "opt_stress_frac",
        "opt_cluster",
        "opt_cluster_k",
        "opt_cluster_maxk",
        "opt_shap",
        "opt_shap_bg",
        "opt_shap_test",
        "opt_vifnorm",
        # Correlation settings
        "opt_corr_method",
        "opt_corr_cap",
        "opt_corr_thr",
        # Repro seed
        "opt_seed",
        # Model selection
        "mdl_task",
        "mdl_lib",
        "mdl_algo",
        # Thresholds
        "thr_auc",
        "thr_f1",
        "thr_ks",
        # Internal helpers
        "__thr_block__",
        "model_selection",
        "effective_cfg",
    ]
    for k in keys_to_clear:
        st.session_state.pop(k, None)


def tvr_finish(
    section_id: str,
    *,
    report_path: Path | None = None,
    report_bytes: bytes | None = None,
    file_name: str,
    summary: dict | None = None,
    label: str | None = None,
    cfg: dict | None = None,
) -> None:
    """Store the finished report (bytes + metadata) and mark section as 'ready'."""
    tvr_init(section_id)
    if report_bytes is None:
        report_bytes = Path(report_path).read_bytes()
    ts = int(time.time())
    st.session_state[_tvr_key(section_id, "bytes")] = report_bytes
    st.session_state[_tvr_key(section_id, "file")] = file_name
    st.session_state[_tvr_key(section_id, "ts")] = ts
    st.session_state[_tvr_key(section_id, "summary")] = summary or {}
    st.session_state[_tvr_key(section_id, "label")] = label or "Run"
    st.session_state[_tvr_key(section_id, "cfg")] = cfg
    st.session_state[_tvr_key(section_id, "stage")] = "ready"


def tvr_render_ready(
    section_id: str, *, header_text: str = "Refit, Validate & Report", filter_metrics_fn=None
) -> None:
    """Render the 'ready' state UI with download button and reset option."""
    tvr_init(section_id)
    if st.session_state[_tvr_key(section_id, "stage")] != "ready":
        return

    st.subheader(header_text)
    s = st.session_state[_tvr_key(section_id, "summary")] or {}
    # drop PSI-like keys if present
    s = {k: v for k, v in s.items() if "psi" not in k.lower()}
    # keep only task-appropriate metrics
    if filter_metrics_fn:
        s = filter_metrics_fn(s)

    if s:
        st.caption("Summary (last run)")
        st.write({k: (round(v, 2) if isinstance(v, (int, float)) else v) for k, v in s.items()})

    st.download_button(
        "⬇️ Download report",
        data=st.session_state[_tvr_key(section_id, "bytes")],
        file_name=st.session_state[_tvr_key(section_id, "file")]
        or f"tanml_report_{st.session_state[_tvr_key(section_id, 'ts')]}.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        key=f"tvr_dl::{section_id}::{st.session_state[_tvr_key(section_id, 'ts')]}",
    )

    # Keep only the "New model / new run" action
    if st.button("✨ New model / new run", key=f"tvr_new::{section_id}"):
        tvr_reset(section_id)
        st.rerun()


def tvr_render_history(section_id: str, *, title: str = "🗂️ Past runs") -> None:
    """Render history (currently no-op)."""
    return  # no-op


def tvr_store_extras(section_id: str, extras: dict[str, Any]) -> None:
    """Store extras data for a section."""
    st.session_state[_tvr_key(section_id, "extras")] = extras
