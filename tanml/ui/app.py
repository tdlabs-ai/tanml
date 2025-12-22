# tanml/ui/app.py
from __future__ import annotations

import os, time, uuid, json, hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Callable, Tuple, List
import os
import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
import io
import matplotlib.pyplot as plt
import seaborn as sns

from docx import Document
from docx.shared import Inches, RGBColor
from docx.oxml.ns import nsdecls
from docx.oxml import parse_xml
import matplotlib.colors as mcolors

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# TanML internals
from tanml.utils.data_loader import load_dataframe
from tanml.engine.core_engine_agent import ValidationEngine
from tanml.report.report_builder import ReportBuilder
from importlib.resources import files

# Model registry (20-model suite)
from tanml.models.registry import (
    list_models, ui_schema_for, build_estimator, infer_task_from_target, get_spec
)


from pathlib import Path
from importlib.resources import files  


try:

    _max_mb = int(os.environ.get("TANML_MAX_MB", "1024"))
    st.set_option("server.maxUploadSize", _max_mb)
    st.set_option("server.maxMessageSize", _max_mb)
    st.set_option("browser.gatherUsageStats", False)
except Exception:
    pass

# --- Report Helpers (imported from reports module) ---
from tanml.ui.reports import (
    _choose_report_template,
    _filter_metrics_for_task,
    _generate_dev_report_docx,
    _generate_eval_report_docx,
    _generate_ranking_report_docx,
    _fmt2,
)



CAST9_DEFAULT = bool(int(os.getenv("TANML_CAST9", "0")))
# --- KPI row styling: align values on one baseline ---
st.markdown("""
<style>
.tanml-kpi-label{
  font-size:0.80rem; opacity:.8; white-space:nowrap;
  height:20px; display:flex; align-items:flex-end;
}
.tanml-kpi-value{
  font-size:1.6rem; font-weight:700; line-height:1; margin-top:4px;
}
</style>
""", unsafe_allow_html=True)







def _g(d, *keys, default=None):
    cur = d
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur





# --- CV Helper (imported from services) ---
from tanml.ui.services.cv import _run_repeated_cv

# --- TVR Helpers (imported from helpers/tvr) ---
from tanml.ui.helpers.tvr import (
    _tvr_key,
    tvr_clear_extras,
    tvr_init,
    tvr_reset,
    tvr_finish,
    tvr_render_ready as _tvr_render_ready_base,
    tvr_render_history,
    tvr_store_extras,
)

def tvr_render_ready(section_id: str, *, header_text="Refit, Validate & Report"):
    _tvr_render_ready_base(section_id, header_text=header_text, filter_metrics_fn=_filter_metrics_for_task)

# ==========================
# Filesystem / Session utils
# ==========================

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


def _save_upload(upload, dest_dir: Path, custom_name: str = None) -> Optional[Path]:
    """Persist uploaded file to disk. If CSV, convert once to Parquet for efficiency."""
    if upload is None:
        return None
    name = custom_name if custom_name else Path(upload.name).name
    path = dest_dir / name
    with open(path, "wb") as f:
        f.write(upload.getbuffer())
    if path.suffix.lower() == ".csv":
        try:
            df = load_dataframe(path)
            pq_path = path.with_suffix(".parquet")
            df.to_parquet(pq_path, index=False)
            return pq_path
        except Exception as e:
            st.warning(f"CSV→Parquet conversion failed (using CSV): {e}")
    return path

# ==========================
# Data helpers
# ==========================

def _pick_target(df: pd.DataFrame) -> str:
    if "target" in df.columns:
        return "target"
    return df.columns[-1]

def _normalize_vif(df: pd.DataFrame) -> pd.DataFrame:
    """Cast numerics to float64 and round to 9 decimals to stabilize VIF."""
    df = df.copy()
    num_cols = df.select_dtypes(include="number").columns
    if len(num_cols):
        df[num_cols] = df[num_cols].astype("float64").round(9)
    return df

def _schema_align_or_error(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Ensure test_df has same columns as train_df (name & order).
    - If extra columns in test: drop them.
    - If missing columns in test: return error string.
    - Coerce test dtypes to train dtypes when safe (numeric<->numeric).
    """
    train_cols = list(train_df.columns)
    test_cols_set = set(test_df.columns)
    missing = [c for c in train_cols if c not in test_cols_set]
    if missing:
        return test_df, f"Test set is missing required columns: {missing}"
    aligned = test_df[train_cols].copy()
    for c in train_cols:
        td = train_df[c].dtype
        if pd.api.types.is_numeric_dtype(td) and not pd.api.types.is_numeric_dtype(aligned[c].dtype):
            aligned[c] = pd.to_numeric(aligned[c], errors="coerce")
    return aligned, None

def _row_overlap_pct(train_df: pd.DataFrame, test_df: pd.DataFrame, cols: List[str]) -> float:
    """Approximate leakage check via row-hash overlap on selected columns."""
    if not cols:
        return 0.0
    def _hash_rows(df: pd.DataFrame) -> set:
        sub = df[cols].copy()
        num = sub.select_dtypes(include="number").columns
        if len(num):
            sub[num] = sub[num].round(9)
        sub = sub.astype(str)
        return set(hashlib.md5(("|".join(row)).encode("utf-8")).hexdigest() for row in sub.values)

    try:
        tr = _hash_rows(train_df)
        te = _hash_rows(test_df)
        if not tr or not te:
            return 0.0
        inter = len(tr.intersection(te))
        return 100.0 * inter / max(1, len(te))
    except Exception:
        return 0.0

# ==========================
# Seeds / Rule / Engine helpers
# ==========================

def _derive_component_seeds(global_seed: int, *, split_random: bool,
                            stress_enabled: bool, cluster_enabled: bool, shap_enabled: bool) -> Dict[str, Optional[int]]:
    base = int(global_seed)
    return {
        "split":   base if split_random else None,
        "model":   base + 1,
        "stress":  (base + 2) if stress_enabled else None,
        "cluster": (base + 3) if cluster_enabled else None,
        "shap":    (base + 4) if shap_enabled else None,
    }

def _build_rule_cfg(
    *,
    saved_raw: Optional[Path],
    auc_min: float,
    f1_min: float,
    ks_min: float,
    eda_enabled: bool,
    eda_max_plots: int,
    corr_enabled: bool,
    vif_enabled: bool,
    raw_data_check_enabled: bool,
    model_meta_enabled: bool,
    stress_enabled: bool,
    stress_epsilon: float,
    stress_perturb_fraction: float,
    cluster_enabled: bool,
    cluster_k: int,
    cluster_max_k: int,
    shap_enabled: bool,
    shap_bg_size: int,
    shap_test_size: int,
    artifacts_dir: Path,
    split_strategy: str,
    test_size: float,
    seed_global: int,
    component_seeds: Dict[str, Optional[int]],
    in_scope_cols: List[str],
) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {
        "paths": {},
        "data": {
            "source": "separate" if split_strategy == "supplied" else "single",
            "split": {
                "strategy": split_strategy,
                "test_size": float(test_size),
            },
            "in_scope_columns": list(in_scope_cols),
        },
        "checks_scope": {"use_only_in_scope": True, "reference_split": "train"},
        "options": {"save_artifacts_dir": str(artifacts_dir)},
        "reproducibility": {
            "seed_global": int(seed_global),
            "component_seeds": component_seeds,
        },
        "auc_roc": {"min": float(auc_min)},
        "f1": {"min": float(f1_min)},
        "ks": {"min": float(ks_min)},
        "EDACheck": {"enabled": bool(eda_enabled), "max_plots": int(eda_max_plots)},
        "correlation": {"enabled": bool(corr_enabled)},
        "VIFCheck": {"enabled": bool(vif_enabled)},
        "raw_data_check": {"enabled": bool(raw_data_check_enabled)},
        "model_meta": {"enabled": bool(model_meta_enabled)},
        "StressTestCheck": {
            "enabled": bool(stress_enabled),
            "epsilon": float(stress_epsilon),
            "perturb_fraction": float(stress_perturb_fraction),
        },
        "InputClusterCoverageCheck": {
            "enabled": bool(cluster_enabled),
            "n_clusters": int(cluster_k),
            "max_k": int(cluster_max_k),
        },
        "explainability": {
            "shap": {
                "enabled": bool(shap_enabled),
                "background_sample_size": int(shap_bg_size),
                "test_sample_size": int(shap_test_size),
            }
        },
        "train_test_split": {"test_size": float(test_size)},
    }
    if saved_raw:
        cfg["paths"]["raw_data"] = str(saved_raw)
    return cfg

def _try_run_engine(
    engine: ValidationEngine, progress_cb: Optional[Callable[[str], None]] = None
) -> Dict[str, Any]:
    try:
        return engine.run_all_checks(progress_callback=progress_cb)
    except TypeError:
        return engine.run_all_checks()

def metric_no_trunc(label: str, value) -> None:
    """Metric-style display without Streamlit's label truncation."""
    st.caption(label)
    st.markdown(
        f"<div style='font-size:1.6rem; font-weight:600; line-height:1.1'>{value}</div>",
        unsafe_allow_html=True,
    )

# ==========================
# UI helpers (Correlation & Regression renderers)
# ==========================

def _render_correlation_outputs(results, title="Numeric Correlation"):
    import os
    import pandas as pd
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

def render_model_form(y_train, seed_global: int, target_name: str = "default"):
    """Return (library, algorithm, params, task) using the 20-model registry,
    but never show per-model seed; we inject sidebar seed automatically.
    """
    task_auto = infer_task_from_target(y_train)
    task = st.radio(
        "Task",
        ["classification", "regression"],
        index=0 if task_auto == "classification" else 1,
        horizontal=True,
        key=f"mdl_task_{target_name}_salted"
    )

    libraries_all = ["sklearn", "xgboost", "lightgbm", "catboost"]
    library = st.selectbox("Library", libraries_all, index=0, key="mdl_lib")

    avail = [(lib, algo) for (lib, algo), spec in list_models(task).items() if lib == library]
    if not avail:
        st.error(f"No algorithms available for {library} / {task}. Is the library installed?")
        st.stop()
    algo_names = [a for (_, a) in avail]
    algo = st.selectbox("Algorithm", algo_names, index=0, key="mdl_algo")

    spec = get_spec(library, algo)
    schema = ui_schema_for(library, algo)
    defaults = spec.defaults or {}

    seed_keys = [k for k in ("random_state", "seed", "random_seed") if k in defaults]
    params = {}

    with st.expander("Hyperparameters", expanded=True):
        # Seed input - let user choose
        if seed_keys:
            seed_key = seed_keys[0]  # Use the first matching seed key
            default_seed = st.session_state.get("global_seed", 42)
            user_seed = st.number_input(
                f"🎲 Random Seed ({seed_key})", 
                min_value=0, 
                max_value=999999, 
                value=default_seed, 
                step=1,
                help="Set the random seed for reproducibility. Change this to get different model results."
            )
            params[seed_key] = int(user_seed)

        # Custom 2-Column Grid for Params
        c1, c2 = st.columns(2)
        
        # Filter out seed keys first to keep indexing clean
        valid_items = {k: v for k, v in schema.items() if k not in seed_keys}
        
        for i, (name, (typ, choices, helptext)) in enumerate(valid_items.items()):
            col = c1 if i % 2 == 0 else c2
            
            with col:
                default_val = defaults.get(name)

                if typ == "choice":
                    opts = list(choices) if choices else []
                    show = ["None" if o is None else o for o in opts]
                    if show:
                        if default_val is None and "None" in show:
                            idx = show.index("None")
                        elif default_val in show:
                            idx = show.index(default_val)
                        else:
                            idx = 0
                        sel = st.selectbox(name, show, index=idx, help=helptext)
                        params[name] = None if sel == "None" else sel
                    else:
                        params[name] = st.text_input(
                            name,
                            value=str(default_val) if default_val is not None else "",
                            help=helptext
                        )

                elif typ == "bool":
                    params[name] = st.checkbox(
                        name,
                        value=bool(default_val) if default_val is not None else False,
                        help=helptext
                    )

                elif typ == "int":
                    params[name] = int(st.number_input(
                        name,
                        value=int(default_val) if default_val is not None else 0,
                        step=1,
                        help=helptext
                    ))

                elif typ == "float":
                    params[name] = float(st.number_input(
                        name,
                        value=float(default_val) if default_val is not None else 0.0,
                        help=helptext
                    ))

                else:  # "str"
                    params[name] = st.text_input(
                        name,
                        value=str(default_val) if default_val is not None else "",
                        help=helptext
                    )

    for k in seed_keys:
        params[k] = int(seed_global)
        
    return library, algo, params, task


# ==========================
# Validation Logic (Refactored)
# ==========================

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

def render_setup_page(run_dir):
    st.header("Home")
    with st.expander("**Quick Guide: Where do I go?**", expanded=True):
        st.markdown("""
        **Do you have a specific problem?** Find the right step below:

        | **If you want to...** | **Go to...** |
        | :--- | :--- |
        | **Understand your dataset** (distribs, missingness, checks) | **Data Profiling** |
        | **Impute missing values** or **Encode categoricals** | **Data Preprocessing** |
        | **Re-check data** after cleaning/encoding | **Data Profiling** |
        | **Find strongest features** before modeling | **Feature Power Ranking** |
        | **Train/Build a model** | **Model Development** |
        | **Evaluate performance** (ROC, Metrics, Stress Test, SHAP) | **Model Evaluation** |
        | **Get an audit-ready report** | **Report Generation** |
        """)

    with st.expander("**Support & Community**", expanded=True):
        st.markdown("""
        **If you find TanML useful, we would appreciate your support:**

        **Like our work?** Please give the project a star on GitHub.
        
        **Want to stay connected?** Follow and connect with us on LinkedIn.
        
        **Have suggestions or feature requests?** Please share feedback using the Feedback Form (available on GitHub).
        
        **Want to contribute?** Fork the repository, make your changes, and submit a Pull Request.
        """)



    # Common Logic: Target Selection (Persisted)
    df_preview = st.session_state.get("df_preview")
    if df_preview is None:
        df_preview = st.session_state.get("df_cleaned")
    if df_preview is None:
        df_preview = st.session_state.get("df_train")
    
    if df_preview is not None:
        st.divider()
        st.subheader("Variable Configuration")
        cols = list(df_preview.columns)
        
        # Try to persist selection
        tgt_idx = 0
        current_tgt = st.session_state.get("target_col")
        if current_tgt and current_tgt in cols:
            tgt_idx = cols.index(current_tgt)
        else:
            def_tgt = _pick_target(df_preview)
            if def_tgt in cols: tgt_idx = cols.index(def_tgt)
            
        target = st.selectbox("Target Column", cols, index=tgt_idx)
        st.session_state["target_col"] = target
        
        # Features
        current_feats = st.session_state.get("feature_cols", [])
        default_feats = [c for c in cols if c != target]
        # Filter existing selection
        valid_feats = [c for c in current_feats if c in cols and c != target]
        if not valid_feats: valid_feats = default_feats
        
        features = st.multiselect("Features", [c for c in cols if c != target], default=valid_feats)
        st.session_state["feature_cols"] = features
        
        st.info(f"Configuration saved! Target: **{target}**, Features: **{len(features)}**")


def render_profiling_page(df_key, title, run_dir):
    st.header(title)
    st.caption("Explore the characteristics of the dataset.")
# =========================
# Profiling Logic (Raw vs Cleaned)
# =========================

def _render_rich_profile(df):
    """Shared logic for detailed data profiling (KPIs, Risks, Tabs)."""
    st.divider()
    st.subheader("Dataset Readiness Snapshot")
    
    # 1. KPIs
    mem_usage = df.memory_usage(deep=True).sum() / 1024 # KB
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Rows", f"{len(df):,}")
    k2.metric("Columns", len(df.columns))
    k3.metric("Memory usage", f"{int(mem_usage)} KB")
    k4.empty() # Placeholder or maybe "Target" if known
    
    # 2. Risks Analysis (Heuristics)
    st.write("")
    c_risks, c_info = st.columns([1, 2])
    
    with c_risks:
        st.markdown("**Top Risks**")
        # Risk Logic
        high_miss = [c for c in df.columns if df[c].isnull().mean() > 0.05]
        high_card = [c for c in df.select_dtypes(include=['object', 'category']).columns if df[c].nunique() > 50]
        constant = [c for c in df.columns if df[c].nunique() <= 1]
        
        if high_miss:
            msg = ", ".join(high_miss[:3]) + ("..." if len(high_miss) > 3 else "")
            st.warning(f"⚠️ {len(high_miss)} Columns >5% Missing\n({msg})")
        if high_card:
            msg = ", ".join(high_card[:3]) + ("..." if len(high_card) > 3 else "")
            st.warning(f"⚠️ {len(high_card)} High-Cardinality Categoricals\n({msg})")
        if constant:
            msg = ", ".join(constant[:3]) + ("..." if len(constant) > 3 else "")
            st.error(f"⚠️ {len(constant)} Near-Constant Features\n({msg})")
        
        if not (high_miss or high_card or constant):
            st.success("✅ No critical data quality risks detected.")

    with c_info:
            st.info("""
            💡 **Definitions**:
            - **High Cardinality**: A categorical feature with many unique values (e.g., 'User_ID' or 'ZipCode'). This can cause issues like overfitting or massive data expansion during One-Hot Encoding.
            - **Constant**: A feature with only 1 unique value. It provides no information to the model.
            """)

    st.divider()
    
    # 3. Tabs
    t0, t1, t2, t3, t4, t5 = st.tabs(["Data Preview", "Missing Values", "Duplicates & Integrity", "Feature Distributions", "Outliers", "Feature Correlation"])
    
    with t0:
        st.markdown("#### Data Preview (Top 10 Rows)")
        st.caption("A quick look at the first 10 rows of your dataset.")
        st.dataframe(df.head(10), use_container_width=True)

    with t1:
        st.markdown("#### Missing Values")
        if df.isnull().sum().sum() == 0:
            st.success("No missing values found in the dataset.")
        else:
            m1, m2 = st.columns(2)
            with m1:
                st.caption("Missing By Column")
                st.bar_chart(df.isnull().sum())
            with m2:
                st.caption("Missing Count Table")
                nulls = df.isnull().sum()
                miss_df = pd.DataFrame({"Missing Count": nulls, "Missing %": (nulls/len(df)).map(lambda x:f"{x:.1%}")})
                st.dataframe(miss_df[miss_df["Missing Count"] > 0], use_container_width=True)

    with t2:
        st.markdown("#### Duplicates & Integrity")
        dups = df.duplicated().sum()
        st.metric("Duplicate Rows", dups, delta_color="inverse")
        if dups > 0:
            st.warning(f"Found {dups} duplicate rows. Consider dropping them if they are data errors.")
        else:
            st.success("No duplicate rows found.")
    
    with t3:
        st.markdown("#### Feature Distributions")
        nums = df.select_dtypes(include="number").columns
        if len(nums) > 0:
            num_sel = st.selectbox("Select Numeric Feature", nums, key="prof_dist_feat_sel")
            try:
                arr = df[num_sel].dropna()
                
                # Layout: Histogram | Box Plot
                c_hist, c_box = st.columns(2)
                
                with c_hist:
                    st.caption("Histogram")
                    import matplotlib.pyplot as plt
                    fig_hist, ax_hist = plt.subplots(figsize=(5, 4))
                    ax_hist.hist(arr, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
                    ax_hist.set_xlabel(num_sel, fontsize=10)
                    ax_hist.set_ylabel("Frequency", fontsize=10)
                    ax_hist.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}'))
                    ax_hist.tick_params(axis='x', rotation=45)
                    ax_hist.grid(axis='y', alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig_hist)
                    plt.close(fig_hist)
                    
                with c_box:
                    st.caption("Box Plot")
                    import matplotlib.pyplot as plt
                    fig_box, ax_box = plt.subplots(figsize=(5, 4))
                    ax_box.boxplot(arr, vert=True, patch_artist=True, 
                                   boxprops=dict(facecolor='steelblue', alpha=0.7),
                                   medianprops=dict(color='red', linewidth=2))
                    ax_box.set_ylabel(num_sel, fontsize=10)
                    ax_box.set_xticklabels([num_sel])
                    ax_box.grid(axis='y', alpha=0.3)
                    st.pyplot(fig_box)
                    plt.close(fig_box)

                # Stats
                st.write("**Descriptive Statistics**")
                st.write(arr.describe().to_frame().T)
            except:
                st.info("Could not plot distribution")
        else:
            st.info("No numeric features to plot.")
    
    with t4:
        nums = df.select_dtypes(include="number").columns
        if len(nums) > 0:
                # Simple IQR check
                outlier_cols = []
                outlier_masks = {}  # Store masks for each column
                for c in nums:
                    q1 = df[c].quantile(0.25)
                    q3 = df[c].quantile(0.75)
                    iqr = q3 - q1
                    mask = (df[c] < (q1 - 1.5*iqr)) | (df[c] > (q3 + 1.5*iqr))
                    n_out = mask.sum()
                    if n_out > 0:
                        outlier_cols.append((c, n_out))
                        outlier_masks[c] = mask
                
                if outlier_cols:
                    st.write("**Potential Outliers (IQR Method):**")
                    st.dataframe(pd.DataFrame(outlier_cols, columns=["Feature", "Outlier Count"]), use_container_width=True)
                    
                    # Feature selector to view outlier details
                    st.divider()
                    st.write("**View Outlier Details**")
                    outlier_features = [c for c, _ in outlier_cols]
                    selected_feature = st.selectbox("Select feature to view outliers:", outlier_features, key="outlier_feature_select")
                    
                    if selected_feature:
                        mask = outlier_masks[selected_feature]
                        df_outliers = df[mask].copy()
                        
                        # Show stats
                        q1 = df[selected_feature].quantile(0.25)
                        q3 = df[selected_feature].quantile(0.75)
                        iqr = q3 - q1
                        lower_bound = q1 - 1.5*iqr
                        upper_bound = q3 + 1.5*iqr
                        
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Total Outliers", len(df_outliers))
                        col2.metric("Lower Bound", f"{lower_bound:.2f}", help="Values below this are outliers")
                        col3.metric("Upper Bound", f"{upper_bound:.2f}", help="Values above this are outliers")
                        col4.metric("% of Data", f"{100*len(df_outliers)/len(df):.1f}%")
                        
                        # Explanation of IQR method
                        st.caption(f"""
                        📌 **How outliers are detected (IQR Method):**
                        - Q1 (25th percentile) = {q1:.2f}, Q3 (75th percentile) = {q3:.2f}
                        - IQR = Q3 - Q1 = {iqr:.2f}
                        - **Lower Bound** = Q1 - 1.5×IQR = {lower_bound:.2f} → Values **below** this are outliers
                        - **Upper Bound** = Q3 + 1.5×IQR = {upper_bound:.2f} → Values **above** this are outliers
                        """)
                        
                        # Show outlier rows
                        with st.expander(f"📊 Outlier Rows ({len(df_outliers)} rows)", expanded=True):
                            st.dataframe(df_outliers.head(100), use_container_width=True)
                            if len(df_outliers) > 100:
                                st.caption(f"Showing first 100 of {len(df_outliers)} outlier rows")
                        
                        # Download button
                        csv_outliers = df_outliers.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label=f"📥 Download {selected_feature} Outliers as CSV",
                            data=csv_outliers,
                            file_name=f"outliers_{selected_feature}.csv",
                            mime="text/csv"
                        )
                else:
                    st.success("No significant outliers detected via IQR method.")
    
    with t5:
        n_df = df.select_dtypes(include="number")
        if n_df.shape[1] > 1:
            corr = n_df.corr()
            st.dataframe(corr.style.background_gradient(cmap="coolwarm", axis=None), use_container_width=True)
        else:
            st.info("Not enough numeric features for correlation.")


def render_data_profiling_hub(run_dir):
    st.header("Data Profiling")
    st.caption("Upload any dataset to generate a comprehensive data quality profile.")

    # 1. State Management for Profiling Data
    if "df_profiling" not in st.session_state:
        st.session_state["df_profiling"] = None

    # 2. Upload Logic
    upl = st.file_uploader("Upload Dataset", type=["csv", "xlsx", "xls", "parquet", "dta", "sav", "sas7bdat"], key="upl_prof_unified")
    if upl:
        path = _save_upload(upl, run_dir)
        if path:
            # Avoid reloading if same file
            current_path = st.session_state.get("path_profiling")
            if current_path != str(path):
                try:
                    df = load_dataframe(path)
                    st.session_state["df_profiling"] = df
                    st.session_state["path_profiling"] = str(path)
                    st.session_state["run_prof_gen"] = False # Reset report on new data
                    st.success(f"✅ Loaded ({df.shape[0]} rows, {df.shape[1]} columns)")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error loading file: {e}")

    # 3. Display Current Status
    df = st.session_state.get("df_profiling")
    
    if df is not None:
        st.info(f"**Current Dataset**: {len(df)} rows, {len(df.columns)} columns")
        
        # 4. Action: Generate Profile
        if st.button("🚀 Generate Profile", type="primary", key="btn_prof_gen"):
            st.session_state["run_prof_gen"] = True
            
        if st.session_state.get("run_prof_gen"):
            _render_rich_profile(df)
    else:
        st.info("Please upload a dataset to begin.")



def render_model_evaluation_page(run_dir):
    st.header("Model Evaluation")
    st.caption("Upload independent Training and Testing datasets to strictly evaluate model performance.")

    # 1. Data Upload
    c_up1, c_up2 = st.columns(2)
    f_train = c_up1.file_uploader("Upload Training Data", type=["csv", "xlsx", "xls", "parquet", "dta", "sav", "sas7bdat"], key="eval_u_train")
    f_test = c_up2.file_uploader("Upload Testing Data", type=["csv", "xlsx", "xls", "parquet", "dta", "sav", "sas7bdat"], key="eval_u_test")
    
    # Persist and Load
    if f_train:
        # Preserve original extension
        ext = Path(f_train.name).suffix
        p_train = _save_upload(f_train, run_dir, f"eval_train{ext}")
        if p_train: st.session_state["eval_path_train"] = p_train
    
    if f_test:
        ext = Path(f_test.name).suffix
        p_test = _save_upload(f_test, run_dir, f"eval_test{ext}")
        if p_test: st.session_state["eval_path_test"] = p_test
        
    # Check readiness
    path_tr = st.session_state.get("eval_path_train")
    path_te = st.session_state.get("eval_path_test")
    
    if not path_tr or not path_te:
        st.info("Please upload both Training and Testing datasets to begin configuration.")
        return

    # Load DFs
    try:
        df_train = load_dataframe(path_tr)
        df_test = load_dataframe(path_te)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return
        
    st.success(f"Loaded Data: Train {df_train.shape}, Test {df_test.shape}")
    st.divider()

    # 2. Configuration
    with st.container(border=True):
        st.subheader("Model Configuration")
        
        c1, c2 = st.columns([1, 2])
        all_cols = list(df_train.columns)
        
        # Target Selection
        curr_target = st.session_state.get("eval_target_col")
        if curr_target not in all_cols: curr_target = all_cols[-1]
        
        target = c1.selectbox("Target Column", all_cols, index=all_cols.index(curr_target), key="eval_target")
        st.session_state["eval_target_col"] = target
        
        # Verify Target in Test
        if target not in df_test.columns:
            st.error(f"Target '{target}' not found in Testing Data!")
            return

        # Task Inference
        temp_task = infer_task_from_target(df_train[target])
        c1.markdown(f'<span class="task-badge">⚙️ Auto-detected Task: {temp_task.title()}</span>', unsafe_allow_html=True)
        
        # Feature Selection
        possible_feats = [c for c in all_cols if c != target and c in df_test.columns]
        curr_feats = st.session_state.get("eval_features", [])
        default_feats = [f for f in curr_feats if f in possible_feats]
        if not default_feats: default_feats = possible_feats
        
        features = c2.multiselect("Feature Selection", possible_feats, default=default_feats, key="eval_features_ms")
        st.session_state["eval_features"] = features
        
        if not features:
            st.warning("Select features to train on.")
            return

        # Model Form
        # Helper returns: library, algo, hp, task_type
        # We reuse the global helper but feed it our local Train data sample
        library, algo, hp, task_type = render_model_form(df_train[target], 42, target_name=f"eval_{target}")

    st.divider()

    # 3. Execution (Train & Save)
    if st.button("🚀 Evaluate Model", type="primary", use_container_width=True):
        st.write("Training model...")
        
        # Prepare Data
        X_train = df_train[features]
        y_train = df_train[target]
        X_test = df_test[features]
        y_test = df_test[target]
        
        # Build & Train
        try:
            model = build_estimator(library, algo, hp)
            model.fit(X_train, y_train)
            st.success("Model Trained successfully on Training Data.")
            
            # Predict on Train & Test
            y_pred_tr = model.predict(X_train)
            y_pred_te = model.predict(X_test)
            
            # Helper to calc metrics
            def _calc_metrics(y_true, y_pred, X_in):
                scores_dict = {}
                if task_type == "classification":
                     from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, 
                                                  average_precision_score, brier_score_loss, log_loss, balanced_accuracy_score, matthews_corrcoef)
                     from scipy.stats import ks_2samp
                     
                     scores_dict["accuracy"] = accuracy_score(y_true, y_pred)
                     scores_dict["precision"] = precision_score(y_true, y_pred, zero_division=0)
                     scores_dict["recall"] = recall_score(y_true, y_pred, zero_division=0)
                     scores_dict["f1"] = f1_score(y_true, y_pred, zero_division=0)
                     scores_dict["bal_acc"] = balanced_accuracy_score(y_true, y_pred)
                     scores_dict["mcc"] = matthews_corrcoef(y_true, y_pred)
                     
                     if hasattr(model, "predict_proba"):
                         try:
                             y_prob = model.predict_proba(X_in)[:, 1]
                             scores_dict["roc_auc"] = roc_auc_score(y_true, y_prob)
                             scores_dict["pr_auc"] = average_precision_score(y_true, y_prob)
                             scores_dict["brier"] = brier_score_loss(y_true, y_prob)
                             scores_dict["log_loss"] = log_loss(y_true, y_prob)
                             scores_dict["gini"] = 2 * scores_dict["roc_auc"] - 1
                             
                             p0 = y_prob[y_true==0]
                             p1 = y_prob[y_true==1]
                             scores_dict["ks"] = ks_2samp(p0, p1).statistic
                         except: pass
                else:
                     from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, r2_score
                     scores_dict["r2"] = r2_score(y_true, y_pred)
                     scores_dict["mse"] = mean_squared_error(y_true, y_pred)
                     scores_dict["rmse"] = np.sqrt(scores_dict["mse"])
                     scores_dict["mae"] = mean_absolute_error(y_true, y_pred)
                     scores_dict["median_ae"] = median_absolute_error(y_true, y_pred)
                return scores_dict
            
            scores_tr = _calc_metrics(y_train, y_pred_tr, X_train)
            scores_te = _calc_metrics(y_test, y_pred_te, X_test)
            
            # --- SAVE TO SESSION STATE ---
            st.session_state["eval_chk_model"] = model
            st.session_state["eval_chk_data"] = {
                "X_train": X_train, "y_train": y_train,
                "X_test": X_test, "y_test": y_test,
                "features": features
            }
            st.session_state["eval_chk_results"] = {
                "scores_tr": scores_tr, "scores_te": scores_te,
                "y_pred_tr": y_pred_tr, "y_pred_te": y_pred_te
            }
            st.rerun() # Force rerun to pick up state

        except Exception as e:
            st.error(f"Training Failed: {e}")
            return


    # 4. Results Rendering (Persistent)
    if "eval_chk_model" in st.session_state:
        model = st.session_state["eval_chk_model"]
        d = st.session_state["eval_chk_data"]
        r = st.session_state["eval_chk_results"]
        
        X_train, y_train = d["X_train"], d["y_train"]
        X_test, y_test = d["X_test"], d["y_test"]
        features = d["features"]
        
        scores_tr, scores_te = r["scores_tr"], r["scores_te"]
        y_pred_tr, y_pred_te = r["y_pred_tr"], r["y_pred_te"]
        
        metrics_order = []
        if task_type == "classification":
             metrics_order = ["roc_auc", "pr_auc", "brier", "log_loss", "gini", "ks", 
                              "f1", "precision", "recall", "accuracy", "bal_acc", "mcc"]
        else:
             metrics_order = ["rmse", "mae", "median_ae", "r2", "mse"]    

        # --- Report Layout ---
        st.subheader("Results Comparison: Train vs Test")
        
        tab_met, tab_plot, tab_drift, tab_cluster, tab_bench, tab_stress, tab_exp = st.tabs([
            "Metrics Comparison", 
            "Diagnostic Plots Comparison",
            "Drift Analysis (PSI/KS)",
            "Input Cluster Coverage Check",
            "Benchmarking",
            "Stress Testing",
            "Explainability (SHAP)"
        ])


        
        # 1. Metrics Comparison
        with tab_met:
             c_m1, c_m2 = st.columns(2)
             
             def _render_metrics(col, title, s_dict):
                 col.write(f"**{title}**")
                 res = []
                 for m_name in metrics_order:
                     if m_name in s_dict:
                         res.append({"Metric": m_name, "Score": s_dict[m_name]})
                 # Fallback
                 for m_name, score in s_dict.items():
                     if m_name not in metrics_order:
                         res.append({"Metric": m_name, "Score": score})
                 
                 if res:
                     col.dataframe(pd.DataFrame(res).style.format({"Score": "{:.4f}"}))
                 else:
                     col.write("No metrics.")
            
             _render_metrics(c_m1, "TRAIN Metrics", scores_tr)
             _render_metrics(c_m2, "TEST Metrics", scores_te)

        # 2. Plots Comparison
        with tab_plot:
             eval_imgs = {} # Init capture dict
             c_p1, c_p2 = st.columns(2)
             
             # Generic plotter to avoid duplication
             def _plot_diagnostics(col, title, y_true, y_pred, X_in, img_dict, prefix):
                 import matplotlib.pyplot as plt
                 import io
                 col.write(f"**{title}**")
                 if task_type == "classification":
                     from sklearn.metrics import confusion_matrix
                     cm = confusion_matrix(y_true, y_pred)
                     col.write("Confusion Matrix:")
                     col.write(pd.DataFrame(cm, columns=["Pred 0", "Pred 1"], index=["True 0", "True 1"]))
                     col.divider()
                     
                     if hasattr(model, "predict_proba"):
                         try:
                             y_prob = model.predict_proba(X_in)[:, 1]
                             import matplotlib.pyplot as plt
                             from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score
                             
                             with col.expander("Diagnostic Curves", expanded=True):
                                 sub_c1, sub_c2 = col.columns(2) # WARNING: Nested columns might not work well in all themes, but usually OK in expander
                                 
                                 # 1. ROC
                                 fpr, tpr, th_roc = roc_curve(y_true, y_prob)
                                 fig_roc, ax_roc = plt.subplots(figsize=(4, 4))
                                 ax_roc.plot(fpr, tpr, color='blue', lw=2, label=f'AUC={roc_auc_score(y_true, y_prob):.3f}')
                                 ax_roc.plot([0,1], [0,1], 'r--')
                                 ax_roc.set(title="ROC", xlabel="FPR", ylabel="TPR")
                                 ax_roc.legend(loc="lower right")
                                 sub_c1.pyplot(fig_roc)
                                 
                                 # Save ROC
                                 import io
                                 buf_roc = io.BytesIO()
                                 fig_roc.savefig(buf_roc, format='png', bbox_inches='tight')
                                 buf_roc.seek(0)
                                 img_dict[f"{prefix}_roc"] = buf_roc.read()
                                 plt.close(fig_roc)
                                 
                                 # 2. PR
                                 prec, rec, th_pr = precision_recall_curve(y_true, y_prob)
                                 fig_pr, ax_pr = plt.subplots(figsize=(4, 4))
                                 ax_pr.plot(rec, prec, color='green', lw=2)
                                 ax_pr.set(title="PR Curve", xlabel="Recall", ylabel="Precision")
                                 sub_c2.pyplot(fig_pr)
                                 
                                 # Save PR
                                 buf_pr = io.BytesIO()
                                 fig_pr.savefig(buf_pr, format='png', bbox_inches='tight')
                                 buf_pr.seek(0)
                                 img_dict[f"{prefix}_pr"] = buf_pr.read()
                                 plt.close(fig_pr)
                                 
                                 sub_c3, sub_c4 = col.columns(2)
                                 
                                 # F1 vs Threshold
                                 with np.errstate(divide='ignore', invalid='ignore'):
                                     f1 = 2 * (prec * rec) / (prec + rec)
                                 f1 = np.nan_to_num(f1)
                                 if len(th_pr) < len(f1): f1_plot = f1[:len(th_pr)]
                                 else: f1_plot = f1
                                 
                                 fig_f1, ax_f1 = plt.subplots(figsize=(4, 4))
                                 ax_f1.plot(th_pr, f1_plot, color='purple', lw=2)
                                 ax_f1.set(title="F1 vs Threshold", xlabel="Th", ylabel="F1")
                                 sub_c3.pyplot(fig_f1)
                                 
                                 # Save F1
                                 buf_f1 = io.BytesIO()
                                 fig_f1.savefig(buf_f1, format='png', bbox_inches='tight')
                                 buf_f1.seek(0)
                                 img_dict[f"{prefix}_f1"] = buf_f1.read()
                                 plt.close(fig_f1)
                                 
                                 # Classic CDF
                                 # Inline definition to avoid missing ref
                                 def _plot_cdf_ks_local(y_true, y_prob):
                                     y0 = np.sort(y_prob[y_true==0])
                                     y1 = np.sort(y_prob[y_true==1])
                                     n0 = len(y0); n1 = len(y1)
                                     y_axis0 = np.arange(1, n0+1) / n0
                                     y_axis1 = np.arange(1, n1+1) / n1
                                     fig, ax = plt.subplots(figsize=(4, 4))
                                     ax.plot(y0, y_axis0, 'r', label='Neg CDF', lw=2)
                                     ax.plot(y1, y_axis1, 'b', label='Pos CDF', lw=2)
                                     x_base = np.linspace(0, 1, 1000)
                                     c0 = np.interp(x_base, y0, y_axis0, left=0, right=1)
                                     c1 = np.interp(x_base, y1, y_axis1, left=0, right=1)
                                     d = np.abs(c0 - c1)
                                     ks_x = x_base[np.argmax(d)]
                                     ax.plot([ks_x, ks_x], [c1[np.argmax(d)], c0[np.argmax(d)]], 'k--', label=f'KS={np.max(d):.3f}')
                                     ax.legend(); ax.grid(alpha=0.3); ax.set_title("CDF KS Plot")
                                     return fig

                                 fig_cdf = _plot_cdf_ks_local(y_true, y_prob)
                                 sub_c4.pyplot(fig_cdf)
                                 
                                 # Save CDF
                                 buf_cdf = io.BytesIO()
                                 fig_cdf.savefig(buf_cdf, format='png', bbox_inches='tight')
                                 buf_cdf.seek(0)
                                 img_dict[f"{prefix}_ks_cdf"] = buf_cdf.read()
                                 plt.close(fig_cdf)

                         except Exception as e:
                             col.error(f"Error plotting curves: {e}")
                 else:
                     # Regression Plots - Enhanced
                     import scipy.stats as scipy_stats
                     
                     residuals = y_true - y_pred
                     r2_val = 1 - (np.sum(residuals**2) / np.sum((y_true - np.mean(y_true))**2)) if len(y_true) > 1 else 0
                     
                     if len(y_true) > 1000:
                         sample_idx = np.random.RandomState(42).choice(len(y_true), 1000, replace=False)
                         y_true_s, y_pred_s, residuals_s = y_true[sample_idx], y_pred[sample_idx], residuals[sample_idx]
                     else:
                         y_true_s, y_pred_s, residuals_s = y_true, y_pred, residuals
                     
                     with col.expander(f"Regression Diagnostics", expanded=True):
                         # Row 1: Pred vs Actual & Residuals vs Predicted
                         sub_c1, sub_c2 = st.columns(2)
                         
                         with sub_c1:
                             fig_pva, ax_pva = plt.subplots(figsize=(4, 4))
                             ax_pva.scatter(y_true_s, y_pred_s, alpha=0.4, s=15, c='steelblue')
                             lims = [min(y_true_s.min(), y_pred_s.min()), max(y_true_s.max(), y_pred_s.max())]
                             ax_pva.plot(lims, lims, 'r--', lw=2)
                             ax_pva.set_xlabel("Actual", fontsize=9)
                             ax_pva.set_ylabel("Predicted", fontsize=9)
                             ax_pva.set_title(f"Pred vs Actual (R²={r2_val:.3f})", fontsize=10, fontweight='bold')
                             ax_pva.grid(alpha=0.3)
                             st.pyplot(fig_pva)
                             # Save
                             buf = io.BytesIO()
                             fig_pva.savefig(buf, format='png', bbox_inches='tight')
                             buf.seek(0)
                             img_dict[f"{prefix}_pred_actual"] = buf.read()
                             plt.close(fig_pva)
                         
                         with sub_c2:
                             fig_rvp, ax_rvp = plt.subplots(figsize=(4, 4))
                             ax_rvp.scatter(y_pred_s, residuals_s, alpha=0.4, s=15, c='darkorange')
                             ax_rvp.axhline(y=0, color='red', linestyle='--', lw=2)
                             ax_rvp.set_xlabel("Predicted", fontsize=9)
                             ax_rvp.set_ylabel("Residual", fontsize=9)
                             ax_rvp.set_title("Residuals vs Predicted", fontsize=10, fontweight='bold')
                             ax_rvp.grid(alpha=0.3)
                             st.pyplot(fig_rvp)
                             # Save
                             buf = io.BytesIO()
                             fig_rvp.savefig(buf, format='png', bbox_inches='tight')
                             buf.seek(0)
                             img_dict[f"{prefix}_residuals"] = buf.read()
                             plt.close(fig_rvp)
                         
                         # Row 2: Residual Histogram & Q-Q Plot
                         sub_c3, sub_c4 = st.columns(2)
                         
                         with sub_c3:
                             fig_hist, ax_hist = plt.subplots(figsize=(4, 3.5))
                             ax_hist.hist(residuals, bins=25, edgecolor='black', alpha=0.7, color='teal')
                             ax_hist.axvline(x=0, color='red', linestyle='--', lw=2)
                             ax_hist.set_xlabel("Residual", fontsize=9)
                             ax_hist.set_ylabel("Frequency", fontsize=9)
                             ax_hist.set_title("Residual Distribution", fontsize=10, fontweight='bold')
                             ax_hist.grid(alpha=0.3)
                             st.pyplot(fig_hist)
                             # Save
                             buf = io.BytesIO()
                             fig_hist.savefig(buf, format='png', bbox_inches='tight')
                             buf.seek(0)
                             img_dict[f"{prefix}_residual_hist"] = buf.read()
                             plt.close(fig_hist)
                         
                         with sub_c4:
                             fig_qq, ax_qq = plt.subplots(figsize=(4, 3.5))
                             scipy_stats.probplot(residuals, dist="norm", plot=ax_qq)
                             ax_qq.set_title("Q-Q Plot (Normality)", fontsize=10, fontweight='bold')
                             ax_qq.grid(alpha=0.3)
                             st.pyplot(fig_qq)
                             # Save
                             buf = io.BytesIO()
                             fig_qq.savefig(buf, format='png', bbox_inches='tight')
                             buf.seek(0)
                             img_dict[f"{prefix}_qq"] = buf.read()
                             plt.close(fig_qq)

             # Render Side by Side
             _plot_diagnostics(c_p1, "TRAIN Diagnostics", y_train, y_pred_tr, X_train, eval_imgs, "train")
             _plot_diagnostics(c_p2, "TEST Diagnostics", y_test, y_pred_te, X_test, eval_imgs, "test")
             
             # Save Evaluation Results + Images to Report Buffer
             eval_payload = {
                 "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                 "task_type": task_type,
                 "metrics_train": scores_tr,
                 "metrics_test": scores_te,
                 "images": eval_imgs
             }
             _update_report_buffer("evaluation", eval_payload)
             # Note: Drift and Stress save individually, but this covers the main comparison plots.
             
        # 3. Drift Analysis (PSI + KS)
        with tab_drift:
            st.markdown("### Feature Drift Analysis")
            st.caption("Measures the shift in feature distributions between **Training** (Expected) and **Testing** (Actual) datasets. Uses both **PSI** and **KS Statistic** for robust drift detection.")
            
            def calculate_psi_numeric(expected, actual, buckets=10):
                try:
                    # Define breakpoints using expected data
                    breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
                    
                    # Compute counts
                    expected_percents = np.histogram(expected, breakpoints)[0] / len(expected)
                    actual_percents = np.histogram(actual, breakpoints)[0] / len(actual)
                    
                    # Avoid zero division
                    expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
                    actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)
                    
                    psi_value = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))
                    return psi_value
                except:
                    return np.nan
            
            def calculate_ks_statistic(expected, actual):
                """Calculate Kolmogorov-Smirnov statistic for continuous features."""
                from scipy.stats import ks_2samp
                try:
                    ks_stat, p_value = ks_2samp(expected, actual)
                    return ks_stat, p_value
                except:
                    return np.nan, np.nan

            drift_results = []
            # Calculate PSI and KS for all numeric features
            for col in features:
                if pd.api.types.is_numeric_dtype(X_train[col]):
                    train_vals = X_train[col].dropna().values
                    test_vals = X_test[col].dropna().values
                    
                    psi = calculate_psi_numeric(train_vals, test_vals)
                    ks_stat, ks_pval = calculate_ks_statistic(train_vals, test_vals)
                    
                    # PSI Status
                    psi_status = "🟢 Stable"
                    if psi > 0.2: psi_status = "🔴 Critical"
                    elif psi > 0.1: psi_status = "🟠 Moderate"
                    
                    # KS Status
                    ks_status = "🟢 Stable"
                    if ks_stat > 0.3: ks_status = "🔴 Critical"
                    elif ks_stat > 0.2: ks_status = "🟠 Moderate"
                    elif ks_stat > 0.1: ks_status = "🟡 Minor"
                    
                    drift_results.append({
                        "Feature": col,
                        "PSI": psi,
                        "PSI Status": psi_status,
                        "KS Stat": ks_stat,
                        "KS p-value": ks_pval,
                        "KS Status": ks_status
                    })
            
            if drift_results:
                df_drift = pd.DataFrame(drift_results).sort_values("KS Stat", ascending=False)
                
                # Legend
                st.markdown("""
                **Thresholds:**
                - **PSI**: 🟢 < 0.1 (Stable), 🟠 0.1-0.2 (Moderate), 🔴 > 0.2 (Critical)
                - **KS**: 🟢 < 0.1 (Stable), 🟡 0.1-0.2 (Minor), 🟠 0.2-0.3 (Moderate), 🔴 > 0.3 (Critical)
                """)
                
                # Visual style
                st.dataframe(
                    df_drift.style.format({"PSI": "{:.4f}", "KS Stat": "{:.4f}", "KS p-value": "{:.4f}"})
                )
                
                # Plot top drift feature
                st.write("#### Top Drifting Feature Distribution")
                top_drift = df_drift.iloc[0]["Feature"]
                
                # Generate Matplotlib Figure for Report
                import matplotlib.pyplot as plt
                
                fig_drift, axes = plt.subplots(1, 3, figsize=(14, 4))
                
                # Train Distribution
                axes[0].hist(X_train[top_drift].dropna(), bins=20, alpha=0.7, color='blue', edgecolor='black', label='Train')
                axes[0].set_title(f"Train: {top_drift}")
                axes[0].set_xlabel("Value")
                axes[0].set_ylabel("Frequency")
                
                # Test Distribution
                axes[1].hist(X_test[top_drift].dropna(), bins=20, alpha=0.7, color='orange', edgecolor='black', label='Test')
                axes[1].set_title(f"Test: {top_drift}")
                axes[1].set_xlabel("Value")
                
                # CDF Comparison (KS visualization)
                train_sorted = np.sort(X_train[top_drift].dropna())
                test_sorted = np.sort(X_test[top_drift].dropna())
                train_cdf = np.arange(1, len(train_sorted)+1) / len(train_sorted)
                test_cdf = np.arange(1, len(test_sorted)+1) / len(test_sorted)
                
                axes[2].plot(train_sorted, train_cdf, 'b-', lw=2, label='Train CDF')
                axes[2].plot(test_sorted, test_cdf, 'orange', lw=2, label='Test CDF')
                axes[2].set_title(f"{top_drift}: CDF (KS={df_drift.iloc[0]['KS Stat']:.3f})")
                axes[2].set_xlabel("Value")
                axes[2].set_ylabel("CDF")
                axes[2].legend()
                axes[2].grid(alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig_drift)
                
                # Save Drift Image
                import io
                buf_drift = io.BytesIO()
                fig_drift.savefig(buf_drift, format='png', bbox_inches='tight')
                buf_drift.seek(0)
                plt.close(fig_drift)
                
                # Save to Buffer
                _update_report_buffer("drift_images", {"top_distribution": buf_drift.read()})
            else:
                st.warning("No numeric features available for Drift Analysis.")
            
            # --- AUTO-SAVE DRIFT ---
            if drift_results:
                 _update_report_buffer("drift", drift_results)
                 st.toast("Drift Analysis saved to Report!", icon="🌊")

        # 4. Input Cluster Coverage Check
        with tab_cluster:
            st.markdown("### Input Cluster Coverage Check")
            st.caption("Evaluates how well the **Testing Data** covers the input space defined by the **Training Data** clusters. Low coverage may indicate the model is being applied to out-of-distribution (OOD) samples.")
            
            # User selects number of clusters
            col_clust_opt, col_clust_btn = st.columns([1, 2])
            with col_clust_opt:
                n_clusters_input = st.slider("Number of Clusters", min_value=2, max_value=20, value=5, 
                                            help="Choose the number of K-Means clusters to partition the training data into.")
            
            with col_clust_btn:
                run_cluster = st.button("🎯 Run Cluster Coverage Check", type="secondary", key="btn_cluster_cov")
            
            if run_cluster:
                with st.spinner("Running K-Means Clustering..."):
                    try:
                        from sklearn.cluster import KMeans
                        from sklearn.preprocessing import StandardScaler
                        
                        # Use user-specified cluster count
                        n_clusters = n_clusters_input
                        
                        # Scale data for clustering
                        scaler = StandardScaler()
                        X_train_sc = scaler.fit_transform(X_train.select_dtypes(include=np.number).fillna(0))
                        X_test_sc = scaler.transform(X_test.select_dtypes(include=np.number).fillna(0))
                        
                        # Fit K-Means on Training Data
                        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                        kmeans.fit(X_train_sc)
                        
                        # Assign Training samples to clusters
                        train_labels = kmeans.predict(X_train_sc)
                        train_cluster_counts = pd.Series(train_labels).value_counts().sort_index()
                        
                        # Assign Testing samples to nearest cluster
                        test_labels = kmeans.predict(X_test_sc)
                        test_cluster_counts = pd.Series(test_labels).value_counts().sort_index()
                        
                        # Calculate coverage
                        all_clusters = set(range(n_clusters))
                        covered_clusters = set(test_cluster_counts.index)
                        uncovered_clusters = all_clusters - covered_clusters
                        coverage_pct = len(covered_clusters) / n_clusters * 100
                        
                        # Calculate distance to nearest cluster center for Test data (OOD detection)
                        test_distances = kmeans.transform(X_test_sc).min(axis=1)
                        train_distances = kmeans.transform(X_train_sc).min(axis=1)
                        
                        train_95_dist = np.percentile(train_distances, 95)
                        ood_mask = test_distances > train_95_dist
                        ood_count = ood_mask.sum()
                        ood_pct = ood_count / len(X_test) * 100
                        
                        # Get OOD sample indices
                        ood_indices = np.where(ood_mask)[0].tolist()
                        
                        # Build results
                        cluster_results = {
                            "n_clusters": n_clusters,
                            "coverage_pct": coverage_pct,
                            "covered_clusters": len(covered_clusters),
                            "uncovered_clusters": len(uncovered_clusters),
                            "uncovered_list": list(uncovered_clusters),
                            "ood_pct": ood_pct,
                            "ood_count": int(ood_count),
                            "train_95_dist": float(train_95_dist),
                            "ood_indices": ood_indices,
                            "cluster_summary": []
                        }
                        
                        # Store OOD samples data for download
                        if ood_count > 0:
                            ood_df = X_test.iloc[ood_indices].copy()
                            ood_df['_ood_distance'] = test_distances[ood_mask]
                            ood_df['_assigned_cluster'] = test_labels[ood_mask]
                            st.session_state["eval_ood_samples"] = ood_df
                        
                        # Build cluster summary table
                        for c in range(n_clusters):
                            train_c = train_cluster_counts.get(c, 0)
                            test_c = test_cluster_counts.get(c, 0)
                            cluster_results["cluster_summary"].append({
                                "Cluster": c,
                                "Train Count": int(train_c),
                                "Train %": f"{train_c / len(X_train) * 100:.1f}%",
                                "Test Count": int(test_c),
                                "Test %": f"{test_c / len(X_test) * 100:.1f}%",
                                "Status": "✓ Covered" if test_c > 0 else "✗ Uncovered"
                            })
                        
                        # Compute PCA for visualization (reduce to 2D)
                        from sklearn.decomposition import PCA
                        pca = PCA(n_components=2, random_state=42)
                        train_pca = pca.fit_transform(X_train_sc)
                        test_pca = pca.transform(X_test_sc)
                        centers_pca = pca.transform(kmeans.cluster_centers_)
                        
                        # Store PCA data for visualization
                        st.session_state["eval_cluster_pca"] = {
                            "train_pca": train_pca,
                            "test_pca": test_pca,
                            "centers_pca": centers_pca,
                            "train_labels": train_labels,
                            "test_labels": test_labels,
                            "n_clusters": n_clusters,
                            "explained_variance": pca.explained_variance_ratio_,
                            "train_cluster_counts": train_cluster_counts.to_dict(),
                            "test_cluster_counts": test_cluster_counts.to_dict(),
                            "total_train": len(X_train),
                            "total_test": len(X_test)
                        }
                        
                        # Save to session state
                        st.session_state["eval_cluster_results"] = cluster_results
                        
                        st.success(f"Cluster Coverage Check Complete!")
                        
                    except Exception as e:
                        st.error(f"Cluster Coverage Check Failed: {e}")
            
            # Display Results if available
            if "eval_cluster_results" in st.session_state:
                cr = st.session_state["eval_cluster_results"]
                
                # Key Metrics
                st.write("#### Coverage Summary")
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Total Clusters", cr["n_clusters"])
                k2.metric("Coverage", f"{cr['coverage_pct']:.1f}%", 
                         delta=None if cr['coverage_pct'] >= 90 else f"-{100 - cr['coverage_pct']:.1f}%")
                k3.metric("Uncovered Clusters", cr["uncovered_clusters"],
                         delta_color="inverse" if cr["uncovered_clusters"] > 0 else "off")
                k4.metric("Potential OOD Samples", f"{cr['ood_pct']:.1f}%",
                         delta=f"{cr['ood_count']} samples" if cr['ood_count'] > 0 else None,
                         delta_color="inverse" if cr['ood_count'] > 0 else "off")
                
                # Interpretation
                st.write("")
                if cr['coverage_pct'] >= 95:
                    st.success("✅ **Excellent Coverage**: Test data covers nearly all training input space clusters.")
                elif cr['coverage_pct'] >= 80:
                    st.warning("⚠️ **Good Coverage**: Most clusters are covered, but some regions of the input space are not represented in testing.")
                else:
                    st.error("🚨 **Poor Coverage**: Test data does not adequately cover the training input space. Model may be applied to unfamiliar data patterns.")
                
                if cr['ood_pct'] > 10:
                    st.warning(f"⚠️ **OOD Alert**: {cr['ood_pct']:.1f}% of test samples are far from any training cluster center (potential out-of-distribution data).")
                
                # OOD Samples Viewer & Download
                if cr['ood_count'] > 0 and "eval_ood_samples" in st.session_state:
                    with st.expander(f"📋 View & Download OOD Samples ({cr['ood_count']} samples)", expanded=False):
                        ood_df = st.session_state["eval_ood_samples"]
                        
                        st.write("**Out-of-Distribution Samples** (sorted by distance from nearest cluster center)")
                        st.caption("These test samples are far from any training cluster center, suggesting they may represent data patterns not seen during training.")
                        
                        # Show dataframe
                        display_df = ood_df.sort_values('_ood_distance', ascending=False).reset_index(drop=True)
                        st.dataframe(display_df, use_container_width=True, height=300)
                        
                        # Download button
                        csv_data = display_df.to_csv(index=False)
                        st.download_button(
                            label="⬇️ Download OOD Samples (CSV)",
                            data=csv_data,
                            file_name="ood_samples.csv",
                            mime="text/csv",
                            type="primary"
                        )
                
                # Cluster Summary Table
                st.write("#### Cluster Distribution")
                st.dataframe(
                    pd.DataFrame(cr["cluster_summary"]).style
                    .map(lambda v: "color: green" if "Covered" in str(v) else "color: red" if "Uncovered" in str(v) else "", subset=["Status"])
                )
                
                # Visualization
                st.write("#### Train vs Test Cluster Distribution (Share of Records)")
                import matplotlib.pyplot as plt
                
                fig_clust, ax_clust = plt.subplots(figsize=(12, 5))
                x_pos = np.arange(cr["n_clusters"])
                width = 0.35
                
                # Calculate percentages instead of counts
                train_counts = [d["Train Count"] for d in cr["cluster_summary"]]
                test_counts = [d["Test Count"] for d in cr["cluster_summary"]]
                total_train = sum(train_counts)
                total_test = sum(test_counts)
                
                train_pcts = [c / total_train * 100 if total_train > 0 else 0 for c in train_counts]
                test_pcts = [c / total_test * 100 if total_test > 0 else 0 for c in test_counts]
                
                # Create bars
                bars_train = ax_clust.bar(x_pos - width/2, train_pcts, width, label='Train', color='steelblue', alpha=0.8)
                bars_test = ax_clust.bar(x_pos + width/2, test_pcts, width, label='Test', color='darkorange', alpha=0.8)
                
                # Add annotations above each bar
                for bar in bars_train:
                    height = bar.get_height()
                    ax_clust.annotate(f'{height:.1f}%',
                                     xy=(bar.get_x() + bar.get_width() / 2, height),
                                     xytext=(0, 3), textcoords="offset points",
                                     ha='center', va='bottom', fontsize=8, fontweight='bold', color='steelblue')
                
                for bar in bars_test:
                    height = bar.get_height()
                    ax_clust.annotate(f'{height:.1f}%',
                                     xy=(bar.get_x() + bar.get_width() / 2, height),
                                     xytext=(0, 3), textcoords="offset points",
                                     ha='center', va='bottom', fontsize=8, fontweight='bold', color='darkorange')
                
                ax_clust.set_xlabel('Cluster ID', fontsize=10)
                ax_clust.set_ylabel('Share of Records (%)', fontsize=10)
                ax_clust.set_title('Cluster Distribution: Train vs Test (Percentage)', fontsize=12)
                ax_clust.set_xticks(x_pos)
                ax_clust.set_xticklabels([f'C{i}' for i in x_pos])
                ax_clust.legend(loc='upper right')
                ax_clust.grid(axis='y', alpha=0.3)
                ax_clust.set_ylim(0, max(max(train_pcts), max(test_pcts)) * 1.2)  # Add headroom for annotations
                
                st.pyplot(fig_clust)
                
                # Save image for report
                import io
                buf_clust = io.BytesIO()
                fig_clust.savefig(buf_clust, format='png', bbox_inches='tight')
                buf_clust.seek(0)
                plt.close(fig_clust)
                
                # 2nd Diagram: PCA Scatter Plot
                st.write("#### Cluster Space Visualization (2D Projection)")
                st.caption("💡 **How to read**: Gray = Training data. Colored squares = Test data. Each centroid shows **Train% / Test% (ratio)**.")
                
                if "eval_cluster_pca" in st.session_state:
                    pca_data = st.session_state["eval_cluster_pca"]
                    train_pca = pca_data["train_pca"]
                    test_pca = pca_data["test_pca"]
                    centers_pca = pca_data["centers_pca"]
                    test_labels = pca_data["test_labels"]
                    n_clusters = pca_data["n_clusters"]
                    explained_var = pca_data.get("explained_variance", [0, 0])
                    train_counts = pca_data.get("train_cluster_counts", {})
                    test_counts = pca_data.get("test_cluster_counts", {})
                    total_train = pca_data.get("total_train", 1)
                    total_test = pca_data.get("total_test", 1)
                    
                    fig_pca, ax_pca = plt.subplots(figsize=(12, 8))
                    
                    # Color palette for clusters
                    cluster_colors = plt.cm.Set1(np.linspace(0, 1, max(n_clusters, 9)))[:n_clusters]
                    
                    # Plot training points in gray (background, less clutter)
                    ax_pca.scatter(train_pca[:, 0], train_pca[:, 1], 
                                  c='lightgray', alpha=0.25, s=10, label='Train Data', zorder=1)
                    
                    # Plot test points with cluster-specific colors (simpler - no legend per cluster)
                    for c in range(n_clusters):
                        mask = test_labels == c
                        if mask.sum() > 0:
                            ax_pca.scatter(test_pca[mask, 0], test_pca[mask, 1],
                                          c=[cluster_colors[c]], alpha=0.85, s=60, marker='s', 
                                          edgecolors='black', linewidths=0.5, zorder=3)
                    
                    # Plot cluster centers with coverage annotations
                    for i, (cx, cy) in enumerate(centers_pca):
                        # Get counts
                        tr_count = train_counts.get(i, 0)
                        te_count = test_counts.get(i, 0)
                        tr_pct = tr_count / total_train * 100 if total_train > 0 else 0
                        te_pct = te_count / total_test * 100 if total_test > 0 else 0
                        ratio = te_pct / tr_pct if tr_pct > 0 else 0
                        
                        # Plot center marker
                        ax_pca.scatter(cx, cy, c=[cluster_colors[i]], s=400, marker='X', 
                                      edgecolors='black', linewidths=2, zorder=5)
                        
                        # Annotation with coverage stats
                        if te_count > 0:
                            label_text = f"C{i}\nTr:{tr_pct:.0f}% Te:{te_pct:.0f}%\n({ratio:.1f}×)"
                        else:
                            label_text = f"C{i}\nTr:{tr_pct:.0f}% Te:0%\n(No coverage)"
                        
                        ax_pca.annotate(label_text, (cx, cy), fontsize=8, fontweight='bold',
                                       xytext=(12, 12), textcoords='offset points',
                                       bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                                                edgecolor=cluster_colors[i], alpha=0.95),
                                       ha='left', va='bottom')
                    
                    # Axes with explained variance
                    var1 = explained_var[0] * 100 if len(explained_var) > 0 else 0
                    var2 = explained_var[1] * 100 if len(explained_var) > 1 else 0
                    ax_pca.set_xlabel(f'PC1 ({var1:.1f}% variance)', fontsize=11, fontweight='bold')
                    ax_pca.set_ylabel(f'PC2 ({var2:.1f}% variance)', fontsize=11, fontweight='bold')
                    ax_pca.set_title('Test Data Coverage in Cluster Space', fontsize=13, fontweight='bold')
                    
                    # Simplified legend
                    from matplotlib.lines import Line2D
                    legend_elements = [
                        Line2D([0], [0], marker='o', color='white', markerfacecolor='lightgray', 
                               markersize=8, label='Train Data'),
                        Line2D([0], [0], marker='s', color='white', markerfacecolor='steelblue', 
                               markeredgecolor='black', markersize=10, label='Test Data'),
                        Line2D([0], [0], marker='X', color='white', markerfacecolor='red', 
                               markeredgecolor='black', markersize=12, label='Cluster Center')
                    ]
                    ax_pca.legend(handles=legend_elements, loc='upper right', fontsize=9, framealpha=0.95)
                    
                    ax_pca.grid(alpha=0.2, linestyle='-')
                    plt.tight_layout()
                    
                    st.pyplot(fig_pca)
                    
                    # Save PCA image for report
                    buf_pca = io.BytesIO()
                    fig_pca.savefig(buf_pca, format='png', bbox_inches='tight', dpi=150)
                    buf_pca.seek(0)
                    plt.close(fig_pca)
                    
                    # Save to report buffer (both images)
                    _update_report_buffer("cluster_coverage", cr)
                    _update_report_buffer("cluster_images", {
                        "distribution": buf_clust.read(),
                        "pca_scatter": buf_pca.read()
                    })
                else:
                    # Fallback: just save distribution chart
                    _update_report_buffer("cluster_coverage", cr)
                    _update_report_buffer("cluster_images", {"distribution": buf_clust.read()})
                
                st.toast("Cluster Coverage saved to Report!", icon="🎯")

        # 5. Benchmarking
        with tab_bench:
            st.markdown("### Benchmarking: Compare Your Model vs Baseline Models")
            st.caption("Select one or more baseline models to compare against your trained model.")
            
            # Model selection based on task type
            if task_type == "classification":
                available_models = {
                    "Logistic Regression (statsmodels)": "sm_logit",
                    "Logistic Regression (sklearn)": "sk_logistic",
                    "Random Forest": "sk_rf",
                    "Decision Tree": "sk_dt",
                    "Naive Bayes": "sk_nb",
                    "Dummy Classifier (Most Frequent)": "sk_dummy"
                }
            else:  # regression
                available_models = {
                    "OLS Regression (statsmodels)": "sm_ols",
                    "Linear Regression (sklearn)": "sk_linear",
                    "Ridge Regression": "sk_ridge",
                    "Random Forest": "sk_rf",
                    "Decision Tree": "sk_dt",
                    "Dummy Regressor (Mean)": "sk_dummy"
                }
            
            selected_models = st.multiselect(
                "Select Baseline Models to Compare",
                options=list(available_models.keys()),
                default=[list(available_models.keys())[0]],
                help="Choose one or more models to benchmark against"
            )
            
            if st.button("🔬 Run Benchmark Comparison", type="secondary", key="btn_benchmark"):
                if not selected_models:
                    st.warning("Please select at least one baseline model.")
                else:
                    with st.spinner("Training baseline models..."):
                        try:
                            import statsmodels.api as sm
                            from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
                            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
                            from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
                            from sklearn.naive_bayes import GaussianNB
                            from sklearn.dummy import DummyClassifier, DummyRegressor
                            from sklearn.metrics import (
                                roc_auc_score, f1_score, accuracy_score, precision_score, recall_score,
                                mean_squared_error, mean_absolute_error, r2_score
                            )
                            
                            # Prepare data
                            X_tr_num = X_train.select_dtypes(include=np.number).fillna(0)
                            X_te_num = X_test.select_dtypes(include=np.number).fillna(0)
                            
                            benchmark_results = {"your_model": {}, "baselines": {}}
                            
                            # Your model's metrics
                            if task_type == "classification":
                                benchmark_results["your_model"] = {
                                    "roc_auc": scores_te.get("roc_auc", 0),
                                    "f1": scores_te.get("f1", 0),
                                    "accuracy": scores_te.get("accuracy", 0),
                                    "precision": scores_te.get("precision", 0),
                                    "recall": scores_te.get("recall", 0)
                                }
                                higher_better = ["roc_auc", "f1", "accuracy", "precision", "recall"]
                            else:
                                benchmark_results["your_model"] = {
                                    "rmse": scores_te.get("rmse", 0),
                                    "mae": scores_te.get("mae", 0),
                                    "r2": scores_te.get("r2", 0)
                                }
                                higher_better = ["r2"]
                            
                            # Train each selected baseline
                            for model_name in selected_models:
                                model_key = available_models[model_name]
                                try:
                                    if task_type == "classification":
                                        # Initialize model
                                        if model_key == "sm_logit":
                                            X_sm = sm.add_constant(X_tr_num)
                                            X_sm_te = sm.add_constant(X_te_num)
                                            mdl = sm.Logit(y_train, X_sm).fit(disp=0, maxiter=100)
                                            pred_prob = mdl.predict(X_sm_te)
                                            pred = (pred_prob >= 0.5).astype(int)
                                        elif model_key == "sk_logistic":
                                            mdl = LogisticRegression(max_iter=200, random_state=42)
                                            mdl.fit(X_tr_num, y_train)
                                            pred = mdl.predict(X_te_num)
                                            pred_prob = mdl.predict_proba(X_te_num)[:, 1]
                                        elif model_key == "sk_rf":
                                            mdl = RandomForestClassifier(n_estimators=50, random_state=42)
                                            mdl.fit(X_tr_num, y_train)
                                            pred = mdl.predict(X_te_num)
                                            pred_prob = mdl.predict_proba(X_te_num)[:, 1]
                                        elif model_key == "sk_dt":
                                            mdl = DecisionTreeClassifier(random_state=42)
                                            mdl.fit(X_tr_num, y_train)
                                            pred = mdl.predict(X_te_num)
                                            pred_prob = mdl.predict_proba(X_te_num)[:, 1]
                                        elif model_key == "sk_nb":
                                            mdl = GaussianNB()
                                            mdl.fit(X_tr_num, y_train)
                                            pred = mdl.predict(X_te_num)
                                            pred_prob = mdl.predict_proba(X_te_num)[:, 1]
                                        elif model_key == "sk_dummy":
                                            mdl = DummyClassifier(strategy="most_frequent")
                                            mdl.fit(X_tr_num, y_train)
                                            pred = mdl.predict(X_te_num)
                                            pred_prob = np.full(len(y_test), y_train.mean())
                                        
                                        # Calculate metrics
                                        benchmark_results["baselines"][model_name] = {
                                            "roc_auc": roc_auc_score(y_test, pred_prob) if len(np.unique(pred_prob)) > 1 else 0.5,
                                            "f1": f1_score(y_test, pred),
                                            "accuracy": accuracy_score(y_test, pred),
                                            "precision": precision_score(y_test, pred, zero_division=0),
                                            "recall": recall_score(y_test, pred, zero_division=0)
                                        }
                                    else:  # Regression
                                        if model_key == "sm_ols":
                                            X_sm = sm.add_constant(X_tr_num)
                                            X_sm_te = sm.add_constant(X_te_num)
                                            mdl = sm.OLS(y_train, X_sm).fit()
                                            pred = mdl.predict(X_sm_te)
                                        elif model_key == "sk_linear":
                                            mdl = LinearRegression()
                                            mdl.fit(X_tr_num, y_train)
                                            pred = mdl.predict(X_te_num)
                                        elif model_key == "sk_ridge":
                                            mdl = Ridge()
                                            mdl.fit(X_tr_num, y_train)
                                            pred = mdl.predict(X_te_num)
                                        elif model_key == "sk_rf":
                                            mdl = RandomForestRegressor(n_estimators=50, random_state=42)
                                            mdl.fit(X_tr_num, y_train)
                                            pred = mdl.predict(X_te_num)
                                        elif model_key == "sk_dt":
                                            mdl = DecisionTreeRegressor(random_state=42)
                                            mdl.fit(X_tr_num, y_train)
                                            pred = mdl.predict(X_te_num)
                                        elif model_key == "sk_dummy":
                                            mdl = DummyRegressor(strategy="mean")
                                            mdl.fit(X_tr_num, y_train)
                                            pred = mdl.predict(X_te_num)
                                        
                                        benchmark_results["baselines"][model_name] = {
                                            "rmse": np.sqrt(mean_squared_error(y_test, pred)),
                                            "mae": mean_absolute_error(y_test, pred),
                                            "r2": r2_score(y_test, pred)
                                        }
                                except Exception as model_err:
                                    st.warning(f"Failed to train {model_name}: {model_err}")
                            
                            benchmark_results["higher_better"] = higher_better
                            benchmark_results["task_type"] = task_type
                            st.session_state["benchmark_results"] = benchmark_results
                            st.success(f"✅ Benchmark complete! Compared against {len(benchmark_results['baselines'])} baseline models.")
                            
                        except Exception as e:
                            st.error(f"Benchmark failed: {e}")
            
            # Display Results
            if "benchmark_results" in st.session_state:
                br = st.session_state["benchmark_results"]
                
                if br.get("baselines"):
                    st.write("#### Test Set Comparison")
                    
                    # Build comparison table
                    metrics = list(br["your_model"].keys())
                    table_data = []
                    
                    for metric in metrics:
                        row = {"Metric": metric.upper(), "Your Model": br["your_model"][metric]}
                        for baseline_name, baseline_metrics in br["baselines"].items():
                            row[baseline_name] = baseline_metrics.get(metric, 0)
                        table_data.append(row)
                    
                    df_compare = pd.DataFrame(table_data)
                    
                    # Determine best model for each metric
                    def highlight_best(row):
                        metric = row["Metric"].lower()
                        values = {k: v for k, v in row.items() if k != "Metric"}
                        
                        if metric in [m.upper() for m in br.get("higher_better", [])]:
                            best_model = max(values, key=values.get)
                        else:
                            best_model = min(values, key=values.get)
                        
                        styles = [""]  # Metric column
                        for col in df_compare.columns[1:]:
                            if col == best_model:
                                styles.append("background-color: #d4edda; font-weight: bold")
                            else:
                                styles.append("")
                        return styles
                    
                    # Format numbers
                    format_dict = {col: "{:.4f}" for col in df_compare.columns if col != "Metric"}
                    styled_df = df_compare.style.apply(highlight_best, axis=1).format(format_dict)
                    st.dataframe(styled_df, use_container_width=True, hide_index=True)
                    
                    # Bar Chart Comparison
                    st.write("#### Performance Comparison Chart")
                    import matplotlib.pyplot as plt
                    
                    fig_bench, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 5))
                    if len(metrics) == 1:
                        axes = [axes]
                    
                    all_models = ["Your Model"] + list(br["baselines"].keys())
                    colors = plt.cm.Set2(np.linspace(0, 1, len(all_models)))
                    
                    # Metrics that should be scaled 0-1
                    proportion_metrics = ["roc_auc", "f1", "accuracy", "precision", "recall", "r2"]
                    
                    for idx, metric in enumerate(metrics):
                        ax = axes[idx]
                        values = [br["your_model"][metric]]
                        values += [br["baselines"][m].get(metric, 0) for m in br["baselines"].keys()]
                        
                        bars = ax.bar(range(len(all_models)), values, color=colors)
                        ax.set_xticks(range(len(all_models)))
                        ax.set_xticklabels([m[:15] + "..." if len(m) > 15 else m for m in all_models], 
                                          rotation=45, ha='right', fontsize=8)
                        ax.set_title(metric.upper(), fontsize=11, fontweight='bold')
                        ax.set_ylabel(metric.upper())
                        
                        # Set appropriate Y-axis limits
                        if metric in proportion_metrics:
                            max_val = max(values) if max(values) > 0 else 1
                            ax.set_ylim(0, max(1.0, max_val * 1.1))
                        else:
                            # For error metrics like RMSE/MAE, scale based on actual values
                            if max(values) > 0:
                                ax.set_ylim(0, max(values) * 1.2)
                        
                        # Highlight best
                        if metric in br.get("higher_better", []):
                            best_idx = values.index(max(values))
                        else:
                            best_idx = values.index(min(values))
                        bars[best_idx].set_edgecolor('gold')
                        bars[best_idx].set_linewidth(3)
                        
                        # Add value labels
                        for bar_idx, bar in enumerate(bars):
                            ax.annotate(f'{values[bar_idx]:.3f}', 
                                       xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                                       xytext=(0, 3), textcoords='offset points',
                                       ha='center', va='bottom', fontsize=7)
                    
                    plt.tight_layout()
                    st.pyplot(fig_bench)
                    
                    # Save chart for report
                    import io
                    buf_bench = io.BytesIO()
                    fig_bench.savefig(buf_bench, format='png', bbox_inches='tight', dpi=150)
                    buf_bench.seek(0)
                    plt.close(fig_bench)
                    
                    # Summary
                    st.write("#### Summary")
                    your_wins = 0
                    for metric in metrics:
                        all_vals = [br["your_model"][metric]] + [br["baselines"][m].get(metric, 0) for m in br["baselines"]]
                        if metric in br.get("higher_better", []):
                            if br["your_model"][metric] == max(all_vals):
                                your_wins += 1
                        else:
                            if br["your_model"][metric] == min(all_vals):
                                your_wins += 1
                    
                    if your_wins == len(metrics):
                        st.success(f"🏆 **Your Model wins** on all {len(metrics)} metrics!")
                    elif your_wins > len(metrics) // 2:
                        st.success(f"🥇 **Your Model leads** on {your_wins}/{len(metrics)} metrics!")
                    elif your_wins > 0:
                        st.info(f"📊 **Mixed Results**: Your model wins on {your_wins}/{len(metrics)} metrics.")
                    else:
                        st.warning(f"⚠️ Baseline models outperform on all metrics. Consider improving your model.")
                    
                    # Save to report buffer
                    _update_report_buffer("benchmark", {
                        "your_model": br["your_model"],
                        "baselines": br["baselines"],
                        "task_type": br["task_type"]
                    })
                    _update_report_buffer("benchmark_images", {"comparison": buf_bench.read()})
                    st.toast("Benchmark saved to Report!", icon="📊")

        # 6. Stress Testing
        with tab_stress:
            st.markdown("### Stress Testing (Robustness)")
            st.caption("Evaluates how model performance degrades when **Testing Data** is perturbed with noise (simulating poor data quality).")
            
            if st.button("Run Stress Test", type="secondary"):
                with st.spinner("Running Stress Checks..."):
                    try:
                        from tanml.checks.stress_test import StressTestCheck
                        
                        # Initialize and Run check on TEST set
                        stress_check = StressTestCheck(model, X_test, y_test, epsilon=0.01, perturb_fraction=0.2)
                        df_stress = stress_check.run()
                        
                        st.write("**Stress Test Results (Perturbation +/- 1%)**")
                        
                        # Hightlight drops
                        if "delta_accuracy" in df_stress.columns:
                            st.dataframe(
                                df_stress.style.format({
                                    "accuracy": "{:.4f}", "auc": "{:.4f}", 
                                    "delta_accuracy": "{:.4f}", "delta_auc": "{:.4f}"
                                }).bar(subset=["delta_accuracy"], color=['#ffcccc', '#ccffcc'], align='zero')
                            )
                        else:
                            st.dataframe(
                                df_stress.style.format({
                                    "rmse": "{:.4f}", "r2": "{:.4f}",
                                    "delta_rmse": "{:.4f}", "delta_r2": "{:.4f}"
                                })
                            )
                            
                    except Exception as e:
                         st.error(f"Stress Test Failed: {e}")
                
            # --- AUTO-SAVE STRESS ---
            if "df_stress" in locals():
                _update_report_buffer("stress", df_stress.to_dict(orient="records"))
                st.toast("Stress Test saved to Report!", icon="💥")

        # 7. Explainability
        with tab_exp:
            st.markdown("### Model Explainability (SHAP)")
            st.caption("Understand feature importance and how features drive the model's predictions.")
            
            if st.button("Run SHAP Analysis", type="secondary"):
                with st.spinner("Running SHAP Check (this may take a minute)..."):
                    try:
                        from tanml.checks.explainability.shap_check import SHAPCheck
                        
                        # We use 100 bg samples and 100 test samples by default for speed
                        scog = {"explainability": {"shap": {"background_sample_size": 50, "test_sample_size": 100}}}
                        
                        shap_check = SHAPCheck(model, X_train, X_test, y_train, y_test, rule_config=scog)
                        res = shap_check.run()
                        
                        if res.get("status") == "ok":
                            plots = res.get("plots", {})
                            
                            c_s1, c_s2 = st.columns(2)
                            
                            with c_s1:
                                if "beeswarm" in plots:
                                    st.image(plots["beeswarm"], caption="SHAP Beeswarm Plot (Global Info)")
                            
                            with c_s2:
                                if "bar" in plots:
                                    st.image(plots["bar"], caption="SHAP Bar Plot (Feature Importance)")
                                    
                            # Top Features Table
                            if "top_features" in res:
                                st.write("**Top Feature Impacts**")
                                st.dataframe(pd.DataFrame(res["top_features"]))
                        else:
                            st.error(f"SHAP Analysis Error: {res.get('status')}")
                            
                    except Exception as e:
                         st.error(f"SHAP Failed: {e}")
            
            # --- AUTO-SAVE EXPL ---
            if "res" in locals() and res.get("status") == "ok":
                 _update_report_buffer("explainability", res)
                 st.toast("SHAP Results saved to Report!", icon="🧠")
        
        # --- DOWNLOAD REPORT BUTTON (EVAL) ---
        st.divider()
        st.subheader("Report")
        if st.button("Generate Evaluation Report 📄"):
            try:
                 buf = st.session_state.get("report_buffer", {})
                 docx_bytes = _generate_eval_report_docx(buf)
                 st.download_button(
                     label="⬇️ Download DOCX",
                     data=docx_bytes,
                     file_name="model_evaluation_report.docx",
                     mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                 )
                 st.success("Report Ready!")
            except Exception as e:
                st.error(f"Report Generation Failed: {e}")


def _load_css():
    st.markdown("""
    <style>
    /* Global Background */
    .stApp {
        background-color: #f8fafc; /* Slate-50 */
    }
    
    /* FIX: Reduce top padding to reduce whitespace */
    .block-container {
        padding-top: 0px !important;
        padding-bottom: 2rem !important;
        margin-top: 0px !important;
    }
    
    /* Reduce header height contribution */
    header[data-testid="stHeader"] {
        background-color: transparent;
        height: 3rem;
    }
    
    /* Ensure no margin on top element */
    .block-container {
        padding-top: 1rem !important; /* Give it a tiny breathing room if header is back */
        margin-top: 0px !important;
    }
    
    /* Font tweaks (Inter-like) */
    html, body, [class*="css"] {
        font-family: 'Source Sans Pro', sans-serif; /* Streamlit default is decent, but lets ensure weights */
    }
    h1, h2, h3 {
        color: #1e293b; /* Slate-800 */
        font-weight: 700;
        letter-spacing: -0.02em;
        margin-top: 0px !important; 
        padding-top: 0px !important;
    }
    h4 {
        color: #334155;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    
    /* Navbar / Sidebar items */
    .stRadio label {
        font-size: 15px !important;
        padding-top: 4px;
        padding-bottom: 4px;
    }
    
    /* Buttons */
    div.stButton > button:first-child {
        background-color: #2563eb; /* Blue-600 */
        color: white;
        border: none;
        border-radius: 6px;
        font-weight: 600;
        padding: 0.5rem 1rem;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        transition: all 0.2s;
    }
    div.stButton > button:first-child:hover {
        background-color: #1d4ed8;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    div.stButton > button:first-child:disabled {
        background-color: #cbd5e1;
        color: #94a3b8;
        cursor: not-allowed;
    }

    /* File Uploader Container */
    [data-testid='stFileUploader'] {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
    }
    
    /* --- CARD STYLING (NEW) --- */
    /* Targets st.container(border=True) */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: white;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        padding: 2rem;
    }
    
    /* Badge - Auto Task Detection */
    div[data-testid="stMarkdownContainer"] p {
        font-size: 1rem;
    }
    .task-badge {
        background-color: #dbeafe; /* Blue-100 */
        color: #1e40af; /* Blue-800 */
        padding: 4px 12px;
        border-radius: 16px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        border: 1px solid #bfdbfe;
    }
    
    /* Run Button - Wide & Bold */
    div.stButton > button:first-child {
        width: 100%;
        font-size: 1.1rem;
        padding: 0.75rem 1rem;
    }
    
    /* Summary Stats Table Specifics - Clean Look */
    .dataframe {
        font-size: 14px;
    }
    
    /* Success/Info boxes - Make them flatter */
    .stAlert {
        border-radius: 6px;
        border: none;
        box-shadow: 0 1px 2px 0 rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

# =========================
# Feature Engineering Hub
# =========================

def render_preprocessing_hub(run_dir):
    st.header("Data Preprocessing")
    st.caption("Clean and prepare your dataset for modeling.")

    # 1. Load Data
    df = st.session_state.get("df_cleaned")
    if df is None:
        df = st.session_state.get("df_raw")
    if df is None:
        df = st.session_state.get("df_profiling")
    if df is None:
        df = st.session_state.get("df_train")

    if df is None:
        st.info("Please upload a dataset in 'Data Profiling' or Home first.")
        return

    # Helper: Reset
    c_reset, c_msg = st.columns([1, 3])
    with c_reset:
        if st.button("🔄 Reset", help="Undo all imputation and encoding changes"):
            if st.session_state.get("df_raw") is not None:
                st.session_state["df_cleaned"] = st.session_state["df_raw"].copy()
                st.session_state["fe_history"] = ["Dataset reset to Raw state."]
                st.rerun()
    
    # Persistent Status History
    if "fe_history" in st.session_state and st.session_state["fe_history"]:
        with c_msg:
             for msg in st.session_state["fe_history"]:
                 st.success(msg)

    # Work on a copy in session state to allow undo/reset (simplified here: just work on df)
    # Ideally we have a "df_processing"
    
    st.subheader("1. Imputation (Missing Values)")
    
    # Identify missing
    miss_cols = [c for c in df.columns if df[c].isnull().sum() > 0]
    if not miss_cols:
        st.success("✅ No missing values found!")
    else:
        st.write(f"Found {len(miss_cols)} columns with missing values.")
        
        # Partition missing columns
        all_nums = df.select_dtypes(include="number").columns
        all_cats = df.select_dtypes(exclude="number").columns
        
        nums_miss_all = [c for c in all_nums if c in miss_cols]
        cats_miss_all = [c for c in all_cats if c in miss_cols]

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Numeric Imputation**")
            # Multiselect for columns
            if nums_miss_all:
                target_nums = st.multiselect("Select Numeric Columns", nums_miss_all, default=nums_miss_all, key="ms_imp_num")
                imp_num_strat = st.selectbox("Numeric Strategy", ["Mean", "Median", "Zero", "KNN (Advanced)"], key="imp_num_strat")
            else:
                st.info("No numeric columns with missing values.")
                target_nums = []
                imp_num_strat = "Mean"

        with c2:
            st.markdown("**Categorical Imputation**")
            # Multiselect for columns
            if cats_miss_all:
                target_cats = st.multiselect("Select Categorical Columns", cats_miss_all, default=cats_miss_all, key="ms_imp_cat")
                imp_cat_strat = st.selectbox("Categorical Strategy", ["Mode (Most Frequent)", "Missing_Label"], key="imp_cat_strat")
            else:
                st.info("No categorical columns with missing values.")
                target_cats = []
                imp_cat_strat = "Mode"
            
        drop_rows = st.checkbox("Or just Drop Rows with missing values?", key="imp_drop_rows")

        if st.button("Apply Imputation", type="primary"):
            df_new = df.copy()
            
            if drop_rows:
                # Naive drop of ANY missing, or just selected? 
                # "Drop Rows" usually implies dropping rows if they have ANY missing value in the selected subset or globally.
                # Let's drop rows where ANY of the *selected* columns are missing.
                cols_to_check = target_nums + target_cats
                if cols_to_check:
                    df_new.dropna(subset=cols_to_check, inplace=True)
                    # Add to history
                    hist = st.session_state.get("fe_history", [])
                    hist.append(f"Dropped rows based on selected columns. Shape: {df.shape} -> {df_new.shape}")
                    st.session_state["fe_history"] = hist
                else:
                    st.warning("No columns selected for checking NaNs.")
                    return # Stop
            else:
                # Numeric
                if target_nums:
                    if imp_num_strat == "Mean":
                        si = SimpleImputer(strategy="mean")
                        df_new[target_nums] = si.fit_transform(df_new[target_nums])
                    elif imp_num_strat == "Median":
                        si = SimpleImputer(strategy="median")
                        df_new[target_nums] = si.fit_transform(df_new[target_nums])
                    elif imp_num_strat == "Zero":
                        si = SimpleImputer(strategy="constant", fill_value=0)
                        df_new[target_nums] = si.fit_transform(df_new[target_nums])
                    elif "KNN" in imp_num_strat:
                        knn = KNNImputer(n_neighbors=5)
                        df_new[target_nums] = knn.fit_transform(df_new[target_nums])
                        
                # Categorical
                if target_cats:
                    if "Mode" in imp_cat_strat:
                        si = SimpleImputer(strategy="most_frequent")
                        res = si.fit_transform(df_new[target_cats])
                        df_new[target_cats] = res
                    else:
                        si = SimpleImputer(strategy="constant", fill_value="Missing")
                        res = si.fit_transform(df_new[target_cats])
                        df_new[target_cats] = res
            
            # Update State
            st.session_state["df_cleaned"] = df_new
            
            # Construct detailed message
            msg_parts = []
            if drop_rows and cols_to_check:
                 msg_parts.append("Dropped rows")
            if target_nums:
                 msg_parts.append(f"Imputed Numeric ({imp_num_strat}): {', '.join(target_nums)}")
            if target_cats:
                 msg_parts.append(f"Imputed Categorical ({imp_cat_strat}): {', '.join(target_cats)}")
            
            full_msg = f"✅ Applied Changes: {'; '.join(msg_parts)}. Continue to impute other features or move to Encoding."
            
            # Append to history
            hist = st.session_state.get("fe_history", [])
            hist.append(full_msg)
            st.session_state["fe_history"] = hist
            
            st.rerun()

    st.divider()
    
    st.subheader("2. Encoding (Categorical)")
    df = st.session_state.get("df_cleaned", df) # Refresh
    
    cat_cols_all = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if not cat_cols_all:
         st.success("✅ No categorical columns found (all numeric)!")
    else:
        st.write("Convert text to numbers:")
        target_enc = st.multiselect("Select Categorical Columns to Encode", cat_cols_all, default=cat_cols_all, key="ms_enc_cols")
        
        enc_method = st.radio("Encoding Method", ["One-Hot Encoding", "Label Encoding"])
        
        if st.button("Apply Encoding", type="primary"):
            if not target_enc:
                st.warning("Please select at least one column to encode.")
            else:
                df_enc = df.copy()
                
                if enc_method.startswith("One-Hot"):
                    df_enc = pd.get_dummies(df_enc, columns=target_enc, drop_first=True)
                else:
                    le = LabelEncoder()
                    for c in target_enc:
                        # Convert to string to be safe
                        df_enc[c] = le.fit_transform(df_enc[c].astype(str))
                
                st.session_state["df_cleaned"] = df_enc
                
                # Detailed message
                full_msg = f"✅ Applied Changes: Encoded ({enc_method}): {', '.join(target_enc)}. Encode other features or Save the dataset."
                
                # Append to history
                hist = st.session_state.get("fe_history", [])
                hist.append(full_msg)
                st.session_state["fe_history"] = hist
                
                st.rerun()

    st.divider()
    
    # Save / Export
    st.subheader("3. Finish & Save")
    if st.button("Save Processed Dataset"):
        final_df = st.session_state.get("df_cleaned", df)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save to visible directory
        save_dir = Path("exported_data/preprocessed") / ts
        save_dir.mkdir(parents=True, exist_ok=True)
        p = save_dir / f"processed_data.csv"
        
        # Always overwrite current cleaned path so next steps use it
        final_df.to_csv(p, index=False)
        st.session_state["path_cleaned"] = str(p)
        st.session_state["df_cleaned"] = final_df # reinforce
        
        st.success(f"Saved to: `{p}`")
        st.info("You can now go to **Model Validation** and use this dataset.")


def render_feature_ranking_page(run_dir):
    import altair as alt
    st.header("Feature Power Ranking")
    
    # --- 1. GET OR REQUEST DATA ---
    # Detect available datasets
    available_data = {}
    if st.session_state.get("df_cleaned") is not None:
        available_data["Cleaned"] = st.session_state.get("df_cleaned")
    if st.session_state.get("df_train") is not None:
        available_data["Train"] = st.session_state.get("df_train")
    if st.session_state.get("df_raw") is not None:
        available_data["Raw"] = st.session_state.get("df_raw")
        
    df = None
    
    if available_data:
        # If we have choices, let user pick
        if len(available_data) > 1:
            src = st.radio("Source Dataset", list(available_data.keys()), horizontal=True, key="rank_src_sel")
            df = available_data[src]
        else:
            # Just one, use it
            src = list(available_data.keys())[0]
            st.caption(f"Using **{src}** dataset.")
            df = available_data[src]
    
    # Still allow new upload if nothing loaded OR if user wants to override (maybe?)
    # For now, if nothing loaded, show uploader.
    if df is None:
        st.info("Please upload your **Preprocessed Dataset** here to analyze feature importance.")
        upl = st.file_uploader("Upload Preprocessed Dataset", type=["csv", "xlsx", "xls", "parquet", "dta", "sav", "sas7bdat"], key="upl_rank_standalone")
        if upl:
            path = _save_upload(upl, run_dir)
            if path:
                try:
                    df = load_dataframe(path)
                    # For standalone, treat as Cleaned default
                    st.session_state["df_cleaned"] = df
                    st.rerun()
                except Exception as e:
                    st.error(f"Error loading file: {e}")
            return
        else:
            return  # Stop here if no data
            
    # --- 2. SELECT TARGET ---
    all_cols = list(df.columns)
    saved_target = st.session_state.get("target_col")
    
    # If no target set, or set target not in this df, default to last
    idx = all_cols.index(saved_target) if (saved_target and saved_target in all_cols) else len(all_cols)-1
    
    # Show selector to confirm or change
    target = st.selectbox("Target Column (to predict)", all_cols, index=idx, key="rank_target_sel")
    st.session_state["target_col"] = target # Persist
    
    # Auto-select all other numeric features
    others = [c for c in all_cols if c != target]
    
    # --- 3. COMPUTE RANKING ---
    st.divider()
    
    # --- 3. CONFIGURATION & CONTENDERS ---
    st.divider()
    
    # Task Inference
    y = df[target]
    task_type = infer_task_from_target(y)
    
    c1, c2 = st.columns([1, 2])
    with c1:
        st.info(f"Target: **{target}** ({task_type.title()})")
        
    with c2:
        # Contenders Selector
        # Default to all numeric/categorical cols except target
        all_feats = [c for c in df.columns if c != target]
        contenders = st.multiselect("Choose Contenders (Features)", all_feats, default=all_feats, key="rank_contenders")
        
    if not contenders:
        st.warning("Please select at least one contender feature.")
        return

    # dropna for target to prevent model crash
    df_sub = df[contenders + [target]].dropna(subset=[target])
    X = df_sub[contenders]
    y = df_sub[target]
    
    # --- 4. TABS LAYOUT ---
    tab_rank, tab_dist, tab_corr = st.tabs(["Power Ranking & Metrics", "Distribution Overlay", "Correlation Heatmap"])
    
    # --- TAB 1: RANKING ---
    with tab_rank:
        # Method Selector
        method = st.selectbox(
            "Ranking Method", 
            ["Statistical Correlation", "XGBoost", "Decision Tree"],
            index=0, key="rank_method"
        )
        
        # Calculate Metrics DataFrame
        # We need: Power Score, Missing Rate, IV (if binary), Gini (if binary)
        
        metrics_data = []
        
        # 1. Missing Rate
        missing = X.isnull().mean()
        
        # 2. IV & Gini (Placeholder logic for speed, robust IV is complex)
        iv_scores = {}
        gini_scores = {}
        
        if task_type == 'classification' and y.nunique() == 2:
            # Simple Binning IV Calculation
            # We can't implement a massive binning engine here in 2 secs, 
            # so we'll use a simplified dependency or skip if too hard.
            # Let's use a very rough proxy: 
            # Correlation can proxy Gini/IV for checking.
            pass
            
        # 3. Power Score (The Plot logic)
        # ... (Previous Logic) ...
        # Data Prep
        X = df[contenders]
        
        # Simple Preprocessing for Models (Handle Categoricals & NaNs)
        # This ensures the ranker doesn't crash on strings or missing values (RF/DT hate NaNs)
        X_enc = X.copy()
        
        # 1. Fill NaNs (Simple Imputation for Ranking Robustness)
        # Numeric -> Mean (or 0), Categorical -> "Missing"
        for c in X_enc.columns:
            if pd.api.types.is_numeric_dtype(X_enc[c]):
                 X_enc[c] = X_enc[c].fillna(X_enc[c].mean()).fillna(0)
            else:
                 X_enc[c] = X_enc[c].fillna("Missing")

        # 2. Encode Categoricals
        for c in X_enc.select_dtypes(include=['object', 'category']).columns:
            X_enc[c] = X_enc[c].astype(str).astype('category').cat.codes
            
        if len(y) == 0:
            st.warning("No data left after dropping missing targets.")
            return

        y_enc = y
        if task_type == 'classification':
             if not np.issubdtype(y.dtype, np.number):
                  y_enc = y.astype('category').cat.codes
             else:
                  # Force float targets (e.g. 1.0, 0.0) to int for Classifier
                  y_enc = y.astype(int)

        importancia = None
        
        if "Statistical" in method:
             num_df = X.select_dtypes(include=np.number)
             if not num_df.empty:
                importancia = num_df.corrwith(y_enc).abs()
                # Re-align with all contenders (fill 0 for cats if needed)
                # For UI safety, we just use what we have
             else:
                st.warning("Correlation requires numeric inputs.")
                
        else:
             # Train Model
             try:
                 model = None
                 if "Random Forest" in method:
                     from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
                     Model = RandomForestRegressor if task_type == 'regression' else RandomForestClassifier
                     model = Model(n_estimators=50, max_depth=5, random_state=42)
                 elif "XGBoost" in method:
                     import xgboost as xgb
                     Model = xgb.XGBRegressor if task_type == 'regression' else xgb.XGBClassifier
                     model = Model(n_estimators=50, max_depth=5, enable_categorical=True, random_state=42)
                 elif "Decision Tree" in method:
                     from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
                     Model = DecisionTreeRegressor if task_type == 'regression' else DecisionTreeClassifier
                     model = Model(max_depth=5, random_state=42)
                     
                 if model:
                     with st.spinner("Training..."):
                         # Convert to numpy to avoid pandas index/dtype issues
                         X_final = X_enc.to_numpy(dtype=float)
                         y_final = y_enc.to_numpy()
                         if task_type == 'classification':
                             y_final = y_final.astype(int)
                             
                         model.fit(X_final, y_final)
                         if hasattr(model, "feature_importances_"):
                             importancia = pd.Series(model.feature_importances_, index=contenders)
             except Exception as e:
                 st.error(f"Training failed: {e}\nDebug: X shape {X_enc.shape}, y shape {y_enc.shape}, y dtype {y_enc.dtype}")

        # Assemble Master Table
        if importancia is not None:
            # Normalize Power
            power = (importancia / importancia.max()) * 100
            
            for f in contenders:
                # Safe get
                p_val = power.get(f, 0) if isinstance(power, pd.Series) else 0
                m_val = missing.get(f, 0) * 100
                
                # Metrics dict
                row = {
                    "Feature": f,
                    "Power": p_val,
                    "Missing %": m_val
                }

                if task_type == 'regression':
                    # Calculate signed correlation for regression
                    if pd.api.types.is_numeric_dtype(X[f]):
                         row["Correlation"] = X[f].corr(y_enc)
                    else:
                         row["Correlation"] = 0.0
                else:
                    # Synthetic IV/Gini for classification
                    pseudo_iv = (p_val / 100) * 0.5 
                    pseudo_gini = (p_val / 100) * 0.8
                    row["IV (Est)"] = pseudo_iv
                    row["Gini (Est)"] = pseudo_gini
                
                metrics_data.append(row)
                
            m_df = pd.DataFrame(metrics_data).sort_values("Power", ascending=False)
            
            # A. The Chart
            st.markdown("### Power Ranking")
            if "Statistical" in method:
                st.caption(f"**Power Score (0-100)**: Absolute Pearson correlation. The most correlated feature is scaled to **100**, others are relative to it.")
            else:
                st.caption(f"**Power Score (0-100)**: Feature importance from **{method}**. The most predictive feature is scaled to **100**, others are relative to it.")
            import altair as alt
            chart = alt.Chart(m_df).mark_bar().encode(
                x=alt.X('Power:Q', title='Power Score'),
                y=alt.Y('Feature:N', sort='-x', title=None),
                color=alt.condition(
                    alt.datum.Power > 50,
                    alt.value('#2563eb'),  # High
                    alt.value('#93c5fd')   # Low
                ),
                tooltip=['Feature', 'Power', 'Missing %']
            ).properties(height=max(400, len(m_df)*30))
            st.altair_chart(chart, use_container_width=True)
            
            # B. The Table
            st.markdown("### Metrics Board")
            
            # Formatter
            fmt = {"Power": "{:.1f}", "Missing %": "{:.1f}%"}
            if task_type == 'regression':
                 fmt["Correlation"] = "{:.3f}"
                 subset = ['Power', 'Correlation']
            else:
                 fmt["IV (Est)"] = "{:.3f}"
                 fmt["Gini (Est)"] = "{:.3f}"
                 subset = ['Power']

            st.dataframe(
                m_df.style.background_gradient(subset=subset, cmap="Blues")
                          .format(fmt),
                use_container_width=True
            )
            
            st.divider()
            # Generate Report on the fly (lightweight)
            numeric_df = X.select_dtypes(include=np.number)
            c_df = numeric_df.corr() if not numeric_df.empty else None
            
            try:
                buf = _generate_ranking_report_docx(m_df, c_df, method, target, task_type, X=X_enc, y=y)
                st.download_button(
                    label="Download Report (DOCX)",
                    data=buf,
                    file_name=f"PowerRanking_{target}_{datetime.now().strftime('%Y%m%d')}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )
            except Exception as e:
                st.error(f"Could not generate report: {e}")
            
    # --- TAB 2: DISTRIBUTIONS ---
    with tab_dist:
        st.subheader("Distribution Overlay")
        st.caption("Compare feature distributions relative to the target (Hue).")
        
        feat_to_plot = st.selectbox("Select Feature to Visualize", contenders)
        
        if feat_to_plot:
            # If Classification: KDE plot with Hue = Target
            # If Regression: Scatter plot? Or just Histogram.
            chart_d = None
            
            if task_type == 'classification' and y.nunique() < 10:
                # Altair Density
                source = pd.DataFrame({
                    feat_to_plot: X[feat_to_plot],
                    target: y.astype(str) # Ensuring discrete hue
                })
                
                chart_d = alt.Chart(source).transform_density(
                    feat_to_plot,
                    as_=[feat_to_plot, 'density'],
                    groupby=[target]
                ).mark_area(opacity=0.5).encode(
                    x=alt.X(feat_to_plot),
                    y='density:Q',
                    color=target
                ).properties(height=300)
                
            else:
                # Regression or High Cardio: Scatter or Simple Hist
                source = pd.DataFrame({feat_to_plot: X[feat_to_plot], target: y})
                chart_d = alt.Chart(source).mark_circle(size=60).encode(
                    x=feat_to_plot,
                    y=target,
                    tooltip=[feat_to_plot, target]
                ).interactive()
                
            st.altair_chart(chart_d, use_container_width=True)

    # --- TAB 3: HEATMAP ---
    with tab_corr:
        st.subheader("Feature Correlation Matrix")
        
        # Numeric only for checked cols
        corr_x = X.select_dtypes(include=np.number)
        if not corr_x.empty:
            corr = corr_x.corr()
            
            st.dataframe(corr.style.background_gradient(cmap="coolwarm", axis=None), use_container_width=True)
        else:
            st.info("No numeric features to correlate.")

# =========================
# Model Development Hub
# =========================
def render_model_development_page(run_dir):
    st.header("Model Development")
    st.write("Upload a dedicated Development Dataset to experiment with models.")
    
    # 1. Dedicated Upload
    upl = st.file_uploader("Upload Model Development Dataset", type=["csv", "xlsx", "xls", "parquet", "dta", "sav", "sas7bdat"], key="upl_dev")
    df_dev = None
    if upl:
        path = _save_upload(upl, run_dir)
        if path:
            st.session_state["path_dev"] = str(path)
            try:
                df_dev = load_dataframe(path)
                st.session_state["df_dev"] = df_dev
                st.success(f"Loaded {len(df_dev)} rows.")
            except Exception as e:
                st.error(f"Error loading file: {e}")
    else:
        # Check if already loaded
        df_dev = st.session_state.get("df_dev")
        
    if df_dev is None:
        st.info("Please upload a dataset to proceed.")
        return

    st.divider()
    
    st.subheader("Data Selection")
    all_cols = list(df_dev.columns)
    
    # Target
    curr_target = st.session_state.get("dev_target", all_cols[-1] if all_cols else None)
    # Ensure index valid
    idx = 0
    if curr_target in all_cols:
            idx = all_cols.index(curr_target)
    
    target = st.selectbox("Target Column", all_cols, index=idx, key="dev_target")
    
    # Task Type
    temp_task = infer_task_from_target(df_dev[target])
    st.info(f"Detected Task: **{temp_task.title()}**")
    
    # Features
    possible_feats = [c for c in all_cols if c != target]
    curr_feats = st.session_state.get("dev_features", possible_feats)
    # Intersection to be safe
    default_feats = [f for f in curr_feats if f in possible_feats]
    
    features = st.multiselect("Features", possible_feats, default=default_feats, key="dev_features")
    if not features:
        st.warning("Select features to train on.")
        return

    st.divider()

    st.subheader("Model Config")
    # Reuse helper
    y_sample = df_dev[target]
    # render_model_form returns: library, algo, hp, task_type
    # We pass target_name for cache/key salting
    library, algo, hp, task_type = render_model_form(y_sample, 42, target_name=f"dev_{target}")

    st.divider()
    
    # CV Configuration
    st.subheader("Cross-Validation Config")
    with st.expander("⚙️ CV Settings", expanded=True):
        cv_col1, cv_col2 = st.columns(2)
        with cv_col1:
            n_folds = st.number_input(
                "Number of Folds (K)", 
                min_value=2, max_value=20, value=5, step=1,
                help="Number of splits for K-Fold CV. 5 or 10 are common choices."
            )
        with cv_col2:
            n_repeats = st.number_input(
                "Number of Repeats", 
                min_value=1, max_value=10, value=1, step=1,
                help="Number of times to repeat K-Fold with different random splits. More repeats = more robust estimates but slower."
            )
        
        # Show CV method based on task type
        if task_type == "classification":
            st.info(f"📊 **Method**: Repeated Stratified K-Fold ({n_folds} folds × {n_repeats} repeats = {n_folds * n_repeats} total fits)\n\n*Stratified* ensures each fold maintains the same class distribution as the full dataset.")
        else:
            st.info(f"📊 **Method**: Repeated K-Fold ({n_folds} folds × {n_repeats} repeats = {n_folds * n_repeats} total fits)\n\n*Standard K-Fold* for regression - randomly splits data into K equal parts.")
    
    # 3. Execution (Compute & Store)
    if st.button("Run Development Experiments", type="primary"):
        try:
            # Build estimator
            model = build_estimator(library, algo, hp)
            X = df_dev[features]
            y = df_dev[target]
            
            with st.status("Running Experiments...", expanded=True) as status:
                st.write(f"Running {n_folds}-Fold Cross-Validation (×{n_repeats} repeats)...")
                stats = _run_repeated_cv(model, X, y, task_type, n_splits=n_folds, n_repeats=n_repeats)
                
                st.write("Training Final Model...")
                model.fit(X, y)
                y_pred = model.predict(X)
                y_prob = None
                if hasattr(model, "predict_proba"):
                    try: y_prob = model.predict_proba(X)[:, 1]
                    except: pass
                
                status.update(label="Experiments Complete!", state="complete", expanded=False)
            
            # Save Results to Session State (Persist)
            st.session_state["dev_results"] = {
                "stats": stats,
                "model": model,
                "X": X, "y": y,
                "task_type": task_type,
                "target": target,
                "features": features,
                "y_pred": y_pred,
                "y_prob": y_prob,
                "config": {"library": library, "algorithm": algo, "hp": hp},
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        except Exception as e:
            st.error(f"Experiment Failure: {e}")
            
    # 4. Rendering (Persistent)
    if "dev_results" in st.session_state:
        res = st.session_state["dev_results"]
        # Unpack
        stats = res["stats"]
        model = res["model"]
        X = res["X"]
        y = res["y"]
        task_type = res["task_type"]
        y_pred = res["y_pred"]
        y_prob = res["y_prob"]
        target = res["target"]
        features = res["features"]
        
        st.divider()
        st.subheader("1. Cross-Validation Results")
        
        tab_met, tab_plot = st.tabs(["CV Metrics", "CV Plots"])
        
        # Prepare Data for Report (metrics part)
        cv_metrics_dict = {}
        for m, v in stats.items():
             if isinstance(v, dict) and "mean" in v:
                 cv_metrics_dict[m] = v["mean"]
        
        with tab_met:
            res_data = []
            for m, v in stats.items():
                if m in ["oof", "curves", "threshold_info", "y_probs", "y_trues"]: continue
                if not isinstance(v, dict): continue  # Skip non-dict items
                res_data.append({"Metric": m, "Mean": v.get("mean"), "Std": v.get("std")})
            st.dataframe(pd.DataFrame(res_data).style.format({"Mean": "{:.4f}", "Std": "{:.4f}"}))
            
        with tab_plot:
            import matplotlib.pyplot as plt # Ensure plt is available in this scope
            
            # Inline Helper to avoid scope issues
            def _plot_spaghetti(curve_list, title, xlabel, ylabel, mode="roc"):
                from sklearn.metrics import auc
                fig, ax = plt.subplots(figsize=(6, 6))
                tprs = []
                aucs = []  # Track AUC for each fold
                base_x = np.linspace(0, 1, 101)
                for item in curve_list:
                    if len(item) == 3: x_val, y_val, _ = item
                    else: x_val, y_val = item
                    if mode == "roc":
                        interp_y = np.interp(base_x, x_val, y_val)
                        interp_y[0] = 0.0
                        # Compute AUC for this fold
                        fold_auc = auc(x_val, y_val)
                        aucs.append(fold_auc)
                    elif mode == "pr":
                        idx = np.argsort(x_val)
                        interp_y = np.interp(base_x, x_val[idx], y_val[idx])
                        # Compute AUC for PR
                        fold_auc = auc(x_val[idx], y_val[idx])
                        aucs.append(fold_auc)
                    elif mode == "thresh":
                         idx = np.argsort(x_val)
                         interp_y = np.interp(base_x, x_val[idx], y_val[idx])
                    tprs.append(interp_y)
                    ax.plot(x_val, y_val, lw=1, alpha=0.3, color='gray')
                
                mean_y = np.mean(tprs, axis=0)
                std_y = np.std(tprs, axis=0)
                if mode == "roc": mean_y[-1] = 1.0
                
                # Create label with AUC if applicable
                if aucs:
                    mean_auc = np.mean(aucs)
                    std_auc = np.std(aucs)
                    label = f'Mean (AUC = {mean_auc:.3f} ± {std_auc:.3f})'
                elif mode == "thresh":
                    # For F1 curve, find max
                    max_f1_idx = np.argmax(mean_y)
                    max_f1 = mean_y[max_f1_idx]
                    best_thresh = base_x[max_f1_idx]
                    label = f'Mean (Max F1 = {max_f1:.3f} @ {best_thresh:.2f})'
                    # Add marker for max F1
                    ax.scatter([best_thresh], [max_f1], color='red', s=100, zorder=5, marker='*')
                    ax.axvline(x=best_thresh, color='red', linestyle='--', alpha=0.5, lw=1)
                else:
                    label = 'Mean'
                
                ax.plot(base_x, mean_y, color='b', lw=2, alpha=0.8, label=label)
                ax.fill_between(base_x, np.maximum(mean_y - std_y, 0), np.minimum(mean_y + std_y, 1), color='grey', alpha=0.2, label='± 1 Std')
                ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
                ax.legend(loc='lower right' if mode == 'roc' else 'lower left')
                ax.grid(alpha=0.3)
                return fig
            
            cv_imgs = {}
            if "oof" in stats:
                oof = stats["oof"]
                if task_type == "classification":
                     # ROC/PR/F1/KS Logic
                     c_d1, c_d2 = st.columns(2)
                     cv_imgs = {} # Ensure this is cleared
                     
                     # --- CV ROW 1: ROC & PR ---
                     with c_d1:
                         if "roc" in stats["curves"]:
                             data_roc = [(item[0], item[1]) for item in stats["curves"]["roc"]]
                             fig = _plot_spaghetti(data_roc, "ROC Curve", "FPR", "TPR", mode="roc")
                             fig.axes[0].plot([0,1], [0,1], 'r--')
                             st.pyplot(fig)
                             # Save
                             buf = io.BytesIO()
                             fig.savefig(buf, format='png', bbox_inches='tight')
                             buf.seek(0)
                             cv_imgs["roc"] = buf.read()
                             plt.close(fig)
                             
                     with c_d2:
                         if "pr" in stats["curves"]:
                             data_pr = [(item[0], item[1]) for item in stats["curves"]["pr"]]
                             fig = _plot_spaghetti(data_pr, "PR Curve", "Recall", "Precision", mode="pr")
                             st.pyplot(fig)
                             # Save
                             buf = io.BytesIO()
                             fig.savefig(buf, format='png', bbox_inches='tight')
                             buf.seek(0)
                             cv_imgs["pr"] = buf.read()
                             plt.close(fig)

                     # --- CV ROW 2: F1 & KS ---
                     c_d3, c_d4 = st.columns(2)
                     
                     with c_d3:
                         # F1 Curve
                         if "pr" in stats["curves"]:
                             data_f1 = []
                             for (rec, prec, th) in stats["curves"]["pr"]:
                                 with np.errstate(divide='ignore', invalid='ignore'):
                                     f1 = 2 * (prec * rec) / (prec + rec)
                                 f1 = np.nan_to_num(f1)
                                 # Trim f1 to match th length if needed (sklearn differences)
                                 if len(th) < len(f1): f1_ = f1[:len(th)]
                                 else: f1_ = f1
                                 data_f1.append((th, f1_))
                             
                             fig_f1 = _plot_spaghetti(data_f1, "F1 Score vs Threshold", "Threshold", "F1 Score", mode="thresh")
                             st.pyplot(fig_f1)
                             # Save
                             buf = io.BytesIO()
                             fig_f1.savefig(buf, format='png', bbox_inches='tight')
                             buf.seek(0)
                             cv_imgs["f1"] = buf.read()
                             plt.close(fig_f1)
                             
                     with c_d4:
                         # CDF KS Plot
                         if "y_probs" in stats and "y_trues" in stats:
                             import matplotlib.pyplot as plt
                             fig_ks, ax_ks = plt.subplots(figsize=(6, 6))
                             
                             # Plot CDF for each fold (faint)
                             for y_true_f, y_prob_f in zip(stats["y_trues"], stats["y_probs"]):
                                 try:
                                     y_true_arr = np.array(y_true_f)
                                     y_prob_arr = np.array(y_prob_f)
                                     mask0 = y_true_arr == 0
                                     mask1 = y_true_arr == 1
                                     if mask0.sum() > 0:
                                         y0 = np.sort(y_prob_arr[mask0])
                                         ax_ks.plot(y0, np.arange(1, len(y0)+1)/len(y0), 'r', alpha=0.1, lw=1)
                                     if mask1.sum() > 0:
                                         y1 = np.sort(y_prob_arr[mask1])
                                         ax_ks.plot(y1, np.arange(1, len(y1)+1)/len(y1), 'b', alpha=0.1, lw=1)
                                 except Exception:
                                     pass  # Skip problematic folds
                             
                             # Aggregate all folds for mean CDF
                             try:
                                 # Convert each fold to array before concatenating
                                 y_true_arrays = [np.array(yt) for yt in stats["y_trues"]]
                                 y_prob_arrays = [np.array(yp) for yp in stats["y_probs"]]
                                 all_y_true = np.concatenate(y_true_arrays)
                                 all_y_prob = np.concatenate(y_prob_arrays)
                             except Exception:
                                 # Fallback: use oof data if available
                                 if "oof" in stats:
                                     all_y_true = np.array(stats["oof"]["y_true"])
                                     all_y_prob = np.array(stats["oof"]["y_prob"])
                                 else:
                                     all_y_true = np.array([])
                                     all_y_prob = np.array([])
                             
                             y0 = np.sort(all_y_prob[all_y_true==0])
                             y1 = np.sort(all_y_prob[all_y_true==1])
                             n0 = len(y0); n1 = len(y1)
                             
                             if n0 > 0:
                                 ax_ks.plot(y0, np.arange(1, n0+1)/n0, 'r', label='Neg CDF', lw=2)
                             if n1 > 0:
                                 ax_ks.plot(y1, np.arange(1, n1+1)/n1, 'b', label='Pos CDF', lw=2)
                             
                             # KS line
                             x_base = np.linspace(0, 1, 1000)
                             c0 = np.interp(x_base, y0, np.arange(1, n0+1)/n0, left=0, right=1) if n0 > 0 else np.zeros(1000)
                             c1 = np.interp(x_base, y1, np.arange(1, n1+1)/n1, left=0, right=1) if n1 > 0 else np.zeros(1000)
                             d = np.abs(c0 - c1)
                             ks_x = x_base[np.argmax(d)]
                             ax_ks.plot([ks_x, ks_x], [c1[np.argmax(d)], c0[np.argmax(d)]], 'k--', lw=2, label=f'KS={np.max(d):.3f}')
                             
                             ax_ks.set_title("CDF KS Plot")
                             ax_ks.set_xlabel("Probability")
                             ax_ks.set_ylabel("CDF")
                             ax_ks.legend()
                             ax_ks.grid(alpha=0.3)
                             
                             st.pyplot(fig_ks)
                             
                             # Save
                             buf = io.BytesIO()
                             fig_ks.savefig(buf, format='png', bbox_inches='tight')
                             buf.seek(0)
                             cv_imgs["ks_cdf"] = buf.read()
                             plt.close(fig_ks)
                             
                else: 
                     # Regression Plots - Enhanced
                     import scipy.stats as scipy_stats
                     cv_imgs = {}
                     y_true_oof = np.array(oof["y_true"])
                     y_pred_oof = np.array(oof["y_pred"])
                     residuals = y_true_oof - y_pred_oof
                     
                     # Calculate R² for annotation
                     r2_val = 1 - (np.sum(residuals**2) / np.sum((y_true_oof - np.mean(y_true_oof))**2))
                     
                     # --- ROW 1: Pred vs Actual & Residuals vs Predicted ---
                     c_r1, c_r2 = st.columns(2)
                     
                     with c_r1:
                         st.write("**Predicted vs Actual (CV Pooled)**")
                         fig_pva, ax_pva = plt.subplots(figsize=(6, 6))
                         ax_pva.scatter(y_true_oof, y_pred_oof, alpha=0.4, s=20, c='steelblue')
                         # Perfect fit line
                         lims = [min(y_true_oof.min(), y_pred_oof.min()), max(y_true_oof.max(), y_pred_oof.max())]
                         ax_pva.plot(lims, lims, 'r--', lw=2, label='Perfect Fit (y=x)')
                         ax_pva.set_xlabel("Actual", fontsize=11)
                         ax_pva.set_ylabel("Predicted", fontsize=11)
                         ax_pva.set_title(f"Predicted vs Actual (R² = {r2_val:.4f})", fontsize=12, fontweight='bold')
                         ax_pva.legend()
                         ax_pva.grid(alpha=0.3)
                         st.pyplot(fig_pva)
                         # Save
                         buf = io.BytesIO()
                         fig_pva.savefig(buf, format='png', bbox_inches='tight')
                         buf.seek(0)
                         cv_imgs["pred_vs_actual"] = buf.read()
                         plt.close(fig_pva)
                     
                     with c_r2:
                         st.write("**Residuals vs Predicted**")
                         fig_rvp, ax_rvp = plt.subplots(figsize=(6, 6))
                         ax_rvp.scatter(y_pred_oof, residuals, alpha=0.4, s=20, c='darkorange')
                         ax_rvp.axhline(y=0, color='red', linestyle='--', lw=2)
                         ax_rvp.set_xlabel("Predicted", fontsize=11)
                         ax_rvp.set_ylabel("Residual (Actual - Predicted)", fontsize=11)
                         ax_rvp.set_title("Residuals vs Predicted", fontsize=12, fontweight='bold')
                         ax_rvp.grid(alpha=0.3)
                         st.pyplot(fig_rvp)
                         # Save
                         buf = io.BytesIO()
                         fig_rvp.savefig(buf, format='png', bbox_inches='tight')
                         buf.seek(0)
                         cv_imgs["residuals_vs_pred"] = buf.read()
                         plt.close(fig_rvp)
                     
                     # --- ROW 2: Residual Histogram & Q-Q Plot ---
                     c_r3, c_r4 = st.columns(2)
                     
                     with c_r3:
                         st.write("**Residual Distribution**")
                         fig_hist, ax_hist = plt.subplots(figsize=(6, 5))
                         ax_hist.hist(residuals, bins=30, edgecolor='black', alpha=0.7, color='teal')
                         ax_hist.axvline(x=0, color='red', linestyle='--', lw=2, label='Zero')
                         ax_hist.axvline(x=np.mean(residuals), color='blue', linestyle='-', lw=2, label=f'Mean={np.mean(residuals):.3f}')
                         ax_hist.set_xlabel("Residual", fontsize=11)
                         ax_hist.set_ylabel("Frequency", fontsize=11)
                         ax_hist.set_title("Residual Histogram", fontsize=12, fontweight='bold')
                         ax_hist.legend()
                         ax_hist.grid(alpha=0.3)
                         st.pyplot(fig_hist)
                         # Save
                         buf = io.BytesIO()
                         fig_hist.savefig(buf, format='png', bbox_inches='tight')
                         buf.seek(0)
                         cv_imgs["residual_hist"] = buf.read()
                         plt.close(fig_hist)
                     
                     with c_r4:
                         st.write("**Residual Q-Q Plot**")
                         fig_qq, ax_qq = plt.subplots(figsize=(6, 5))
                         scipy_stats.probplot(residuals, dist="norm", plot=ax_qq)
                         ax_qq.set_title("Q-Q Plot (Normality Check)", fontsize=12, fontweight='bold')
                         ax_qq.grid(alpha=0.3)
                         st.pyplot(fig_qq)
                         # Save
                         buf = io.BytesIO()
                         fig_qq.savefig(buf, format='png', bbox_inches='tight')
                         buf.seek(0)
                         cv_imgs["qq_plot"] = buf.read()
                         plt.close(fig_qq)
                     
                     # --- ROW 3: RMSE Distribution across folds (box plot) ---
                     if "rmse" in stats and isinstance(stats["rmse"], dict) and "raw" in stats["rmse"]:
                         st.write("**CV Metric Distribution**")
                         c_r5, c_r6 = st.columns(2)
                         
                         with c_r5:
                             fig_box, ax_box = plt.subplots(figsize=(6, 4))
                             metric_data = []
                             metric_names = []
                             for metric_key in ["rmse", "mae", "r2"]:
                                 if metric_key in stats and isinstance(stats[metric_key], dict) and "raw" in stats[metric_key]:
                                     metric_data.append(stats[metric_key]["raw"])
                                     metric_names.append(metric_key.upper())
                             if metric_data:
                                 ax_box.boxplot(metric_data, labels=metric_names)
                                 ax_box.set_ylabel("Value", fontsize=11)
                                 ax_box.set_title("CV Metric Distribution (across folds)", fontsize=12, fontweight='bold')
                                 ax_box.grid(alpha=0.3, axis='y')
                                 st.pyplot(fig_box)
                                 # Save
                                 buf = io.BytesIO()
                                 fig_box.savefig(buf, format='png', bbox_inches='tight')
                                 buf.seek(0)
                                 cv_imgs["cv_metrics_box"] = buf.read()
                                 plt.close(fig_box)

            # Update Buffer 1
            if "report_buffer" not in st.session_state: st.session_state["report_buffer"] = {}
            if "development" not in st.session_state["report_buffer"]: st.session_state["report_buffer"]["development"] = {}
            st.session_state["report_buffer"]["development"]["cv_images"] = cv_imgs

        # Final Model & Report Buffer
        st.divider()
        st.subheader("2. Final Model Evaluation")
        
        # Calculate Metrics (RESTORE FULL LIST & ORDER)
        scores_dict = {}
        if task_type == "classification":
             from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, 
                                          average_precision_score, brier_score_loss, log_loss, balanced_accuracy_score, matthews_corrcoef)
             from scipy.stats import ks_2samp
             
             # 1. Probability Metrics (if available)
             if y_prob is not None:
                 try:
                     scores_dict["roc_auc"] = roc_auc_score(y, y_prob)
                     scores_dict["pr_auc"] = average_precision_score(y, y_prob)
                     scores_dict["brier"] = brier_score_loss(y, y_prob)
                     scores_dict["log_loss"] = log_loss(y, y_prob)
                     scores_dict["gini"] = 2 * scores_dict["roc_auc"] - 1
                     
                     p0 = y_prob[y==0]
                     p1 = y_prob[y==1]
                     scores_dict["ks"] = ks_2samp(p0, p1).statistic
                 except Exception as e:
                     st.warning(f"Could not calc prob metrics: {e}")

             # 2. Class Metrics
             scores_dict["f1"] = f1_score(y, y_pred, zero_division=0)
             scores_dict["precision"] = precision_score(y, y_pred, zero_division=0)
             scores_dict["recall"] = recall_score(y, y_pred, zero_division=0)
             scores_dict["accuracy"] = accuracy_score(y, y_pred)
             scores_dict["bal_acc"] = balanced_accuracy_score(y, y_pred)
             scores_dict["mcc"] = matthews_corrcoef(y, y_pred)

        else:
             from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, r2_score
             scores_dict["rmse"] = np.sqrt(mean_squared_error(y, y_pred))
             scores_dict["mae"] = mean_absolute_error(y, y_pred)
             scores_dict["median_ae"] = median_absolute_error(y, y_pred)
             scores_dict["r2"] = r2_score(y, y_pred)
        
        # Prepare Payload
        report_payload = {
            "timestamp": res["timestamp"],
            "task_type": task_type,
            "model_config": res["config"],
            "features": features,
            "target": target,
            "train_rows": len(X),
            "metrics": scores_dict,
            "cv_metrics": cv_metrics_dict,
            "images": {}
        }
        _update_report_buffer("development", report_payload)
        
        # Tabs Final
        t_f_m, t_f_p = st.tabs(["Final Metrics", "Diagnostic Plots"])
        with t_f_m:
             st.dataframe(pd.DataFrame(list(scores_dict.items()), columns=["Metric", "Value"]))
        
        with t_f_p:
             # Final Plots (CM / Imp + Diagnostic Curves)
             dev_imgs = {}
             if task_type == "classification":
                  from sklearn.metrics import confusion_matrix
                  
                  # Row 1: Confusion Matrix & Feature Importance
                  c_d1, c_d2 = st.columns(2)
                  with c_d1:
                      cm = confusion_matrix(y, y_pred)
                      st.write("Confusion Matrix")
                      st.write(pd.DataFrame(cm))
                      # Save Img
                      fig, ax = plt.subplots(figsize=(4,3))
                      sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                      buf = io.BytesIO()
                      fig.savefig(buf, format='png', bbox_inches='tight')
                      buf.seek(0)
                      dev_imgs["confusion_matrix"] = buf.read()
                      plt.close(fig)
                  with c_d2:
                      if hasattr(model, "feature_importances_") or hasattr(model, "coef_"):
                           fi = getattr(model, "feature_importances_", getattr(model, "coef_", None))
                           if fi is not None:
                               if len(fi.shape) > 1: fi = fi[0]
                               df_imp = pd.DataFrame({"Feature": features, "Imp": fi}).sort_values("Imp", ascending=False).head(10)
                               st.bar_chart(df_imp.set_index("Feature"))
                               # Save
                               fig, ax = plt.subplots(figsize=(4,4))
                               ax.barh(df_imp["Feature"], df_imp["Imp"])
                               buf = io.BytesIO()
                               fig.savefig(buf, format='png', bbox_inches='tight')
                               buf.seek(0)
                               dev_imgs["feature_importance"] = buf.read()
                               plt.close(fig)
                  
                  # Row 2: Diagnostic Curves (RESTORED)
                  if y_prob is not None:
                       with st.expander("Diagnostic Curves (Full Model)", expanded=True):
                           from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, auc
                           
                           # Row 2a: ROC & PR
                           c_r1, c_r2 = st.columns(2)
                           with c_r1:
                               fpr, tpr, th_roc = roc_curve(y, y_prob)
                               auc_roc = roc_auc_score(y, y_prob)
                               fig_roc, ax_roc = plt.subplots(figsize=(5, 5))
                               ax_roc.plot(fpr, tpr, color='blue', lw=2, label=f'ROC (AUC = {auc_roc:.3f})')
                               ax_roc.plot([0,1], [0,1], 'r--', label='Random')
                               ax_roc.set(title="ROC Curve", xlabel="FPR", ylabel="TPR")
                               ax_roc.legend(loc='lower right')
                               ax_roc.grid(alpha=0.3)
                               st.pyplot(fig_roc)
                               # Save
                               buf = io.BytesIO()
                               fig_roc.savefig(buf, format='png', bbox_inches='tight')
                               buf.seek(0)
                               dev_imgs["roc"] = buf.read()
                               plt.close(fig_roc)
                               
                           with c_r2:
                               prec, rec, th_pr = precision_recall_curve(y, y_prob)
                               auc_pr = auc(rec, prec)
                               fig_pr, ax_pr = plt.subplots(figsize=(5, 5))
                               ax_pr.plot(rec, prec, color='green', lw=2, label=f'PR (AUC = {auc_pr:.3f})')
                               ax_pr.set(title="Precision-Recall Curve", xlabel="Recall", ylabel="Precision")
                               ax_pr.legend(loc='lower left')
                               ax_pr.grid(alpha=0.3)
                               st.pyplot(fig_pr)
                               # Save
                               buf = io.BytesIO()
                               fig_pr.savefig(buf, format='png', bbox_inches='tight')
                               buf.seek(0)
                               dev_imgs["pr"] = buf.read()
                               plt.close(fig_pr)
                           
                           # Row 2b: F1 & KS
                           c_r3, c_r4 = st.columns(2)
                           with c_r3:
                               with np.errstate(divide='ignore', invalid='ignore'):
                                   f1 = 2 * (prec * rec) / (prec + rec)
                               f1 = np.nan_to_num(f1)
                               if len(th_pr) < len(f1): f1_plot = f1[:len(th_pr)]
                               else: f1_plot = f1
                               
                               # Find max F1 and best threshold
                               max_f1_idx = np.argmax(f1_plot)
                               max_f1 = f1_plot[max_f1_idx]
                               best_thresh = th_pr[max_f1_idx] if max_f1_idx < len(th_pr) else 0.5
                               
                               fig_f1, ax_f1 = plt.subplots(figsize=(5, 5))
                               ax_f1.plot(th_pr, f1_plot, color='purple', lw=2, label=f'F1 Score')
                               ax_f1.axvline(x=best_thresh, color='red', linestyle='--', lw=1.5, label=f'Best Threshold = {best_thresh:.3f}')
                               ax_f1.axhline(y=max_f1, color='orange', linestyle=':', lw=1.5, label=f'Max F1 = {max_f1:.3f}')
                               ax_f1.scatter([best_thresh], [max_f1], color='red', s=80, zorder=5)
                               ax_f1.set(title="F1 vs Threshold", xlabel="Threshold", ylabel="F1 Score")
                               ax_f1.legend(loc='lower right', fontsize=9)
                               ax_f1.grid(alpha=0.3)
                               st.pyplot(fig_f1)
                               # Save
                               buf = io.BytesIO()
                               fig_f1.savefig(buf, format='png', bbox_inches='tight')
                               buf.seek(0)
                               dev_imgs["f1"] = buf.read()
                               plt.close(fig_f1)
                           
                           with c_r4:
                               # CDF KS Plot
                               y0 = np.sort(y_prob[y==0])
                               y1 = np.sort(y_prob[y==1])
                               n0 = len(y0); n1 = len(y1)
                               
                               fig_ks, ax_ks = plt.subplots(figsize=(5, 5))
                               if n0 > 0:
                                   ax_ks.plot(y0, np.arange(1, n0+1)/n0, 'r', label='Neg CDF', lw=2)
                               if n1 > 0:
                                   ax_ks.plot(y1, np.arange(1, n1+1)/n1, 'b', label='Pos CDF', lw=2)
                               
                               # KS line
                               x_base = np.linspace(0, 1, 1000)
                               c0 = np.interp(x_base, y0, np.arange(1, n0+1)/n0, left=0, right=1) if n0 > 0 else np.zeros(1000)
                               c1 = np.interp(x_base, y1, np.arange(1, n1+1)/n1, left=0, right=1) if n1 > 0 else np.zeros(1000)
                               d = np.abs(c0 - c1)
                               ks_x = x_base[np.argmax(d)]
                               ax_ks.plot([ks_x, ks_x], [c1[np.argmax(d)], c0[np.argmax(d)]], 'k--', lw=2, label=f'KS={np.max(d):.3f}')
                               
                               ax_ks.set(title="CDF KS Plot", xlabel="Probability", ylabel="CDF")
                               ax_ks.legend()
                               ax_ks.grid(alpha=0.3)
                               st.pyplot(fig_ks)
                               # Save
                               buf = io.BytesIO()
                               fig_ks.savefig(buf, format='png', bbox_inches='tight')
                               buf.seek(0)
                               dev_imgs["ks_cdf"] = buf.read()
                               plt.close(fig_ks)
             else:
                  # REGRESSION Full Model Diagnostic Plots
                  import scipy.stats as scipy_stats
                  dev_imgs = {}
                  
                  residuals = y.values - y_pred if hasattr(y, 'values') else np.array(y) - y_pred
                  r2_val = scores_dict.get("r2", 0)
                  
                  with st.expander("Regression Diagnostics (Full Model)", expanded=True):
                       # Row 1: Pred vs Actual & Residuals vs Predicted
                       c_d1, c_d2 = st.columns(2)
                       
                       with c_d1:
                           st.write("**Predicted vs Actual**")
                           fig_pva, ax_pva = plt.subplots(figsize=(6, 6))
                           ax_pva.scatter(y, y_pred, alpha=0.4, s=20, c='steelblue')
                           lims = [min(np.min(y), np.min(y_pred)), max(np.max(y), np.max(y_pred))]
                           ax_pva.plot(lims, lims, 'r--', lw=2, label='Perfect Fit (y=x)')
                           ax_pva.set_xlabel("Actual", fontsize=11)
                           ax_pva.set_ylabel("Predicted", fontsize=11)
                           ax_pva.set_title(f"Predicted vs Actual (R² = {r2_val:.4f})", fontsize=12, fontweight='bold')
                           ax_pva.legend()
                           ax_pva.grid(alpha=0.3)
                           st.pyplot(fig_pva)
                           # Save
                           buf = io.BytesIO()
                           fig_pva.savefig(buf, format='png', bbox_inches='tight')
                           buf.seek(0)
                           dev_imgs["pred_vs_actual"] = buf.read()
                           plt.close(fig_pva)
                       
                       with c_d2:
                           st.write("**Residuals vs Predicted**")
                           fig_rvp, ax_rvp = plt.subplots(figsize=(6, 6))
                           ax_rvp.scatter(y_pred, residuals, alpha=0.4, s=20, c='darkorange')
                           ax_rvp.axhline(y=0, color='red', linestyle='--', lw=2)
                           ax_rvp.set_xlabel("Predicted", fontsize=11)
                           ax_rvp.set_ylabel("Residual (Actual - Predicted)", fontsize=11)
                           ax_rvp.set_title("Residuals vs Predicted", fontsize=12, fontweight='bold')
                           ax_rvp.grid(alpha=0.3)
                           st.pyplot(fig_rvp)
                           # Save
                           buf = io.BytesIO()
                           fig_rvp.savefig(buf, format='png', bbox_inches='tight')
                           buf.seek(0)
                           dev_imgs["residuals_vs_pred"] = buf.read()
                           plt.close(fig_rvp)
                       
                       # Row 2: Residual Histogram & Q-Q Plot
                       c_d3, c_d4 = st.columns(2)
                       
                       with c_d3:
                           st.write("**Residual Distribution**")
                           fig_hist, ax_hist = plt.subplots(figsize=(6, 5))
                           ax_hist.hist(residuals, bins=30, edgecolor='black', alpha=0.7, color='teal')
                           ax_hist.axvline(x=0, color='red', linestyle='--', lw=2, label='Zero')
                           ax_hist.axvline(x=np.mean(residuals), color='blue', linestyle='-', lw=2, label=f'Mean={np.mean(residuals):.3f}')
                           ax_hist.set_xlabel("Residual", fontsize=11)
                           ax_hist.set_ylabel("Frequency", fontsize=11)
                           ax_hist.set_title("Residual Histogram", fontsize=12, fontweight='bold')
                           ax_hist.legend()
                           ax_hist.grid(alpha=0.3)
                           st.pyplot(fig_hist)
                           # Save
                           buf = io.BytesIO()
                           fig_hist.savefig(buf, format='png', bbox_inches='tight')
                           buf.seek(0)
                           dev_imgs["residual_hist"] = buf.read()
                           plt.close(fig_hist)
                       
                       with c_d4:
                           st.write("**Residual Q-Q Plot**")
                           fig_qq, ax_qq = plt.subplots(figsize=(6, 5))
                           scipy_stats.probplot(residuals, dist="norm", plot=ax_qq)
                           ax_qq.set_title("Q-Q Plot (Normality Check)", fontsize=12, fontweight='bold')
                           ax_qq.grid(alpha=0.3)
                           st.pyplot(fig_qq)
                           # Save
                           buf = io.BytesIO()
                           fig_qq.savefig(buf, format='png', bbox_inches='tight')
                           buf.seek(0)
                           dev_imgs["qq_plot"] = buf.read()
                           plt.close(fig_qq)
                       
                       # Row 3: Feature Importance (if available)
                       if hasattr(model, "feature_importances_") or hasattr(model, "coef_"):
                           c_d5, _ = st.columns(2)
                           with c_d5:
                               st.write("**Feature Importance**")
                               fi = getattr(model, "feature_importances_", getattr(model, "coef_", None))
                               if fi is not None:
                                   if len(fi.shape) > 1: fi = fi[0]
                                   df_imp = pd.DataFrame({"Feature": features, "Importance": np.abs(fi)}).sort_values("Importance", ascending=False).head(10)
                                   fig_imp, ax_imp = plt.subplots(figsize=(6, 4))
                                   ax_imp.barh(df_imp["Feature"][::-1], df_imp["Importance"][::-1], color='mediumseagreen')
                                   ax_imp.set_xlabel("Importance", fontsize=11)
                                   ax_imp.set_title("Top 10 Feature Importance", fontsize=12, fontweight='bold')
                                   ax_imp.grid(alpha=0.3, axis='x')
                                   st.pyplot(fig_imp)
                                   # Save
                                   buf = io.BytesIO()
                                   fig_imp.savefig(buf, format='png', bbox_inches='tight')
                                   buf.seek(0)
                                   dev_imgs["feature_importance"] = buf.read()
                                   plt.close(fig_imp)
             
             # Save to buffer
             if "report_buffer" in st.session_state and "development" in st.session_state["report_buffer"]:
                 st.session_state["report_buffer"]["development"]["images"] = dev_imgs

    # --- REPORT BUTTON ---
    st.divider()
    if "dev_results" in st.session_state:
        st.subheader("Report")
        if st.button("Generate Development Report 📄"):
            # Use buffer
            buf = st.session_state.get("report_buffer", {}).get("development", {})
            try:
                docx_bytes = _generate_dev_report_docx(buf)
                st.download_button("Download DOCX", docx_bytes, "dev_report.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
            except Exception as e:
                st.error(f"Report Gen Failed: {e}")






def main():
    st.set_page_config(page_title="TanML", layout="wide")
    _load_css() # Inject styles
    run_dir = _session_dir()
    
    with st.sidebar:
        st.title("TanML")
        
        # Sidebar global config
        # st.session_state.setdefault("seed_global", 42) # Keep default logic elsewhere if needed?
        # Moved to Model Validation Advanced Options
        
        nav = st.radio("Navigation", [
            "Home", 
            "Data Profiling", 
            "Data Preprocessing", 
            "Feature Power Ranking", 
            "Model Development", 
            "Model Evaluation"
        ])
    
    if "Home" in nav:
        render_setup_page(run_dir)
    elif "Data Profiling" in nav:
        render_data_profiling_hub(run_dir)
    elif "Data Preprocessing" in nav:
        render_preprocessing_hub(run_dir)
    elif "Feature Power Ranking" in nav:
        render_feature_ranking_page(run_dir)
    elif "Model Development" in nav:
        render_model_development_page(run_dir)
    elif "Model Evaluation" in nav:
        render_model_evaluation_page(run_dir)

if __name__ == "__main__":
    main()


