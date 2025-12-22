# tanml/ui/components/renderers.py
"""
Specialized result renderers for TanML UI.
"""

from __future__ import annotations

import os
from pathlib import Path
import streamlit as st
import pandas as pd

from tanml.ui.reports import _fmt2


# Let's move _g to helpers.py (if it exists) or keep local copy. 
# Copying local for safety and simplicity as it is very small.

def _g(d, *keys, default=None):
    cur = d
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur

from tanml.ui.helpers.tvr import _tvr_key

from tanml.ui.components.metrics import metric_no_trunc


def _render_correlation_outputs(results, title="Numeric Correlation"):
    st.subheader(title)
    corr_res = (results or {}).get("CorrelationCheck", {}) or {}
    art = corr_res.get("artifacts", corr_res)
    summary = corr_res.get("summary", {}) or {}
    notes = corr_res.get("notes", []) or []

    heatmap_path = art.get("heatmap_path")
    if heatmap_path and os.path.exists(heatmap_path):
        caption = (
            f"Correlation Heatmap ({summary.get('method','')}) — "
            + ("full matrix" if summary.get('plotted_full_matrix') else
               f"showing {summary.get('plotted_features')}/{summary.get('n_numeric_features')} features (subset)")
        )
        st.image(heatmap_path, caption=caption, width="stretch")
    else:
        st.info("Heatmap not available.")

    # Format the threshold so 0.4 shows as 0.40 (avoid 0.39999999999999997)
    thr_raw = summary.get("threshold", None)
    thr_str = _fmt2(float(thr_raw), decimals=2) if isinstance(thr_raw, (int, float)) else "—"

    c1, c2, c3, c4 = st.columns(4)
    labels = ["Numeric features", "Pairs evaluated", f"Pairs ≥ {thr_str}", "Method"]
    values = [
        summary.get("n_numeric_features", "—"),
        summary.get("n_pairs_total", "—"),
        summary.get("n_pairs_flagged_ge_threshold", "—"),
        (summary.get("method", "—") or "—").capitalize(),
    ]
    for col, lab in zip((c1, c2, c3, c4), labels):
        col.markdown(f'<div class="tanml-kpi-label">{lab}</div>', unsafe_allow_html=True)
    for col, val in zip((c1, c2, c3, c4), values):
        col.markdown(f'<div class="tanml-kpi-value">{val}</div>', unsafe_allow_html=True)

    main_csv = art.get("top_pairs_main_csv")
    if main_csv and os.path.exists(main_csv):
        prev_df = pd.read_csv(main_csv)
        st.caption("Top high-correlation pairs (preview)")
        st.dataframe(prev_df.head(3), width="stretch", height=160, hide_index=True)
    else:
        st.info("No high-correlation pairs at current threshold.")

    full_csv = art.get("top_pairs_csv")
    if full_csv and os.path.exists(full_csv):
        with open(full_csv, "rb") as f:
            st.download_button(
                "⬇️ Download full pairs CSV",
                f,
                file_name="correlation_top_pairs.csv",
                mime="text/csv",
                key=f"corrcsv::{full_csv}"
            )
    for n in notes:
        st.caption(f"• {n}")


def _render_regression_outputs(results):
    """Show regression tiles + detailed metrics only (no visuals)."""
    task = results.get("task_type", "classification")
    if task != "regression":
        return

    # Key tiles
    st.subheader("Key Metrics")
    c1, c2, c3 = st.columns(3)
    c1.metric("RMSE", _fmt2(_g(results, "summary", "rmse")))
    c2.metric("MAE",  _fmt2(_g(results, "summary", "mae")))
    r2_val = _g(results, "summary", "r2")
    c3.metric("R²",   _fmt2(r2_val, decimals=2))

    # Detailed metrics
    st.subheader("Regression Metrics (Detailed)")
    adj_r2 = _g(results, "RegressionMetrics", "r2_adjusted")
    med_ae = _g(results, "RegressionMetrics", "median_ae")
    mv     = _g(results, "RegressionMetrics", "mape_or_smape")
    m_is_mape = bool(_g(results, "RegressionMetrics", "mape_used", default=False))

    st.table({
        "Metric": ["Adjusted R²", "Median AE", "MAPE/SMAPE"],
        "Value": [
            _fmt2(adj_r2, decimals=2) if adj_r2 is not None else "N/A",
            _fmt2(med_ae, decimals=2) if med_ae is not None else "—",
            (f"{_fmt2(mv, decimals=2)}% (MAPE)" if m_is_mape else
             (f"{_fmt2(mv, decimals=2)}% (SMAPE)" if mv is not None else "N/A")),
        ],
    })

    # Notes, if any
    notes = _g(results, "RegressionMetrics", "notes", default=[]) or []
    if notes:
        with st.expander("Metric notes"):
            for n in notes:
                st.write("• " + str(n))


def tvr_render_extras(section_id: str):
    extras = st.session_state.get(_tvr_key(section_id, "extras"))
    if not extras:
        return

    results = extras.get("results", {}) or {}
    task = results.get("task_type", "classification")

    # Correlation always (if present)
    _render_correlation_outputs(results)

    # Task-aware metrics/dashboard
    if task == "regression":
        _render_regression_outputs(results)
    else:
        st.subheader("Key Metrics")
        summary = (results.get("summary") or {})
        auc_val        = _fmt2(summary.get("auc"))
        ks_val         = _fmt2(summary.get("ks"))
        rules_failed_v = _fmt2(summary.get("rules_failed"))
        c1, c2, c3 = st.columns(3)
        c1.metric("AUC", auc_val)
        c2.metric("KS", ks_val)
        c3.metric("Rules failed", rules_failed_v)

    st.subheader("Run Summary")
    st.write({
        "Train rows": int(extras.get("train_rows", 0) or 0),
        "Test rows": int(extras.get("test_rows", 0) or 0),
        "Target": extras.get("target", "—"),
        "Features used": int(extras.get("n_features", 0) or 0),
        "Seed used": extras.get("seed_used", "—"),
        "Effective config": extras.get("eff_path", "—"),
        "Artifacts dir": extras.get("artifacts_dir", "—"),
    })

    # ------- Artifacts -------
    art_dir = Path(extras.get("artifacts_dir") or "")
    if not art_dir or not art_dir.exists():
        return

    # Optional: show first SHAP image (if any)
    shap_imgs = sorted(art_dir.glob("**/*shap*.*"))
    if shap_imgs:
        st.image(str(shap_imgs[0]), caption="SHAP summary", width="stretch")

    st.subheader("Artifacts")

    SKIP_SUFFIXES = ("_top_pairs_main.csv",)
    pretty_names = {
        "correlation_top_pairs.csv": "Flagged pairs (full)",
        "heatmap.png": "Correlation heatmap",
        "pearson_corr.csv": "Pearson correlation matrix",
        "spearman_corr.csv": "Spearman correlation matrix",
    }
    order_hint = [
        "correlation_top_pairs.csv",
        "heatmap.png",
        "pearson_corr.csv",
        "spearman_corr.csv",
    ]

    all_paths = [p for p in art_dir.glob("**/*") if p.is_file()]
    files_list = [p for p in all_paths if not any(p.name.endswith(sfx) for sfx in SKIP_SUFFIXES)]
    if not files_list:
        st.caption("No artifacts were saved.")
        return

    files_list.sort(key=lambda p: (order_hint.index(p.name) if p.name in order_hint else 999, p.name.lower()))

    for p in files_list[:100]:
        label = pretty_names.get(p.name, p.name)
        with open(p, "rb") as fh:
            st.download_button(
                f"⬇️ Download {label}",
                fh.read(),
                file_name=p.name,
                key=f"art::{section_id}::{p}",
                width="stretch",
            )
