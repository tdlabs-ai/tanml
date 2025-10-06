# tanml/ui/app.py
from __future__ import annotations

import os, time, uuid, json, hashlib
from pathlib import Path
from typing import Optional, Dict, Any, Callable, Tuple, List
import os
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split

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

def _choose_report_template(task_type: str) -> Path:
    """Return the correct .docx template for 'regression' or 'classification'."""
    # packaged location (recommended): tanml/report/templates/*.docx
    try:
        templates_pkg = files("tanml.report.templates")
        name = "report_template_reg.docx" if task_type == "regression" else "report_template_cls.docx"
        p = templates_pkg / name
        if p.is_file():
            return Path(str(p))
    except Exception:
        pass

    # repo fallback
    repo_guess = Path(__file__).resolve().parents[1] / "report" / "templates"
    p2 = repo_guess / ("report_template_reg.docx" if task_type == "regression" else "report_template_cls.docx")
    if p2.exists():
        return p2

    # cwd fallback
    return Path.cwd() / ("report_template_reg.docx" if task_type == "regression" else "report_template_cls.docx")


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





def _filter_metrics_for_task(summary: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only metrics relevant to the task_type inside summary."""
    if not isinstance(summary, dict):
        return summary or {}

    task = summary.get("task_type")
    cls_keys = {"auc", "ks", "f1", "pr_auc", "rules_failed", "task_type"}
    reg_keys = {"rmse", "mae", "r2", "rules_failed", "task_type"}  

    if task == "classification":
        return {k: v for k, v in summary.items() if k in cls_keys}
    if task == "regression":
        return {k: v for k, v in summary.items() if k in reg_keys}
    return summary


def _g(d, *keys, default=None):
    cur = d
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur

def _fmt2(v, *, decimals=2, dash="—"):
    if v is None:
        return dash
    try:
        if isinstance(v, float):
            return f"{v:.{decimals}f}"
        if isinstance(v, int):
            return str(v)
        return str(v)
    except Exception:
        return dash

# ---------- TVR (Train/Validate/Report) helpers ----------

def _tvr_key(section_id: str, name: str) -> str:
    return f"tvr::{section_id}::{name}"

def tvr_clear_extras(section_id: str):
    st.session_state.pop(_tvr_key(section_id, "extras"), None)

def tvr_reset(section_id: str):
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
        # Uploaders (single-file flow)
        "upl_cleaned", "upl_raw_single",
        # Uploaders (train/test flow)
        "upl_train", "upl_test", "upl_raw_global",

        # Sidebar options
        "opt_eda", "opt_eda_max",
        "opt_corr", "opt_vif",
        "opt_rawcheck", "opt_modelmeta",
        "opt_stress", "opt_stress_eps", "opt_stress_frac",
        "opt_cluster", "opt_cluster_k", "opt_cluster_maxk",
        "opt_shap", "opt_shap_bg", "opt_shap_test",
        "opt_vifnorm",

        # Correlation settings
        "opt_corr_method", "opt_corr_cap", "opt_corr_thr",

        # Repro seed
        "opt_seed",

        # Model selection
        "mdl_task", "mdl_lib", "mdl_algo",

        # Thresholds
        "thr_auc", "thr_f1", "thr_ks",

        # Internal helpers
        "__thr_block__", "model_selection", "effective_cfg",
    ]
    for k in keys_to_clear:
        st.session_state.pop(k, None)

def tvr_init(section_id: str):
    for k, v in {
        "stage": "idle",
        "bytes": None,
        "file": None,
        "ts": None,
        "summary": None,
        "label": None,
        "cfg": None,
        #"history": [],
    }.items():
        st.session_state.setdefault(_tvr_key(section_id, k), v)

def tvr_finish(section_id: str, *, report_path: Path = None, report_bytes: bytes = None,
               file_name: str, summary: dict = None, label: str = None, cfg: dict = None):
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

    

def tvr_render_ready(section_id: str, *, header_text="Refit, Validate & Report"):
    tvr_init(section_id)
    if st.session_state[_tvr_key(section_id, "stage")] != "ready":
        return

    st.subheader(header_text)
    s = st.session_state[_tvr_key(section_id, "summary")] or {}
    # drop PSI-like keys if present
    s = {k: v for k, v in s.items() if "psi" not in k.lower()}
    # keep only task-appropriate metrics
    s = _filter_metrics_for_task(s)


    if s:
        st.caption("Summary (last run)")
        st.write({k: (round(v, 2) if isinstance(v, (int, float)) else v) for k, v in s.items()})

    st.download_button(
        "⬇️ Download report",
        data=st.session_state[_tvr_key(section_id, "bytes")],
        file_name=st.session_state[_tvr_key(section_id, "file")] or
                  f"tanml_report_{st.session_state[_tvr_key(section_id,'ts')]}.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        key=f"tvr_dl::{section_id}::{st.session_state[_tvr_key(section_id,'ts')]}",
        width="stretch",
    )

    # Keep only the "New model / new run" action
    if st.button("✨ New model / new run", key=f"tvr_new::{section_id}", width="stretch"):
        tvr_reset(section_id)
        st.rerun()

def tvr_render_history(section_id: str, *, title="🗂️ Past runs"):
    return  # no-op
   
def tvr_store_extras(section_id: str, extras: Dict[str, Any]):
    st.session_state[_tvr_key(section_id, "extras")] = extras

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

def _save_upload(upload, dest_dir: Path) -> Optional[Path]:
    """Persist uploaded file to disk. If CSV, convert once to Parquet for efficiency."""
    if upload is None:
        return None
    name = Path(upload.name).name
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

# ==========================
# UI — Refit-only (20 models)
# ==========================

st.set_page_config(page_title="TanML — Refit & Validate", layout="wide")
st.title("TanML • Refit & Validate")

run_dir = _session_dir()
artifacts_dir = run_dir / "artifacts"

# Sidebar

with st.sidebar.expander("Checks & Options", expanded=True):
    eda_enabled = st.checkbox("EDA plots", True, key="opt_eda")
    eda_max_plots = st.number_input("EDA max plots (-1=all numeric)", value=-1, step=1, key="opt_eda_max")
    corr_enabled = st.checkbox("Correlation matrix", True, key="opt_corr")
    vif_enabled = st.checkbox("VIF check", True, key="opt_vif")
    raw_data_check_enabled = st.checkbox("RawDataCheck (needs raw)", True, key="opt_rawcheck")
    model_meta_enabled = st.checkbox("Model metadata", True, key="opt_modelmeta")

with st.sidebar.expander("Robustness / Stress Testing", expanded=True):
    stress_enabled = st.checkbox("StressTestCheck", True, key="opt_stress")
    stress_epsilon = st.number_input("Epsilon (noise)", 0.0, 1.0, 0.01, 0.01, key="opt_stress_eps")
    stress_perturb_fraction = st.number_input("Perturb fraction", 0.0, 1.0, 0.20, 0.05, key="opt_stress_frac")

with st.sidebar.expander("Input Cluster Coverage", expanded=False):
    cluster_enabled = st.checkbox("InputClusterCoverageCheck", True, key="opt_cluster")
    cluster_k = st.number_input("n_clusters", 2, 50, 5, 1, key="opt_cluster_k")
    cluster_max_k = st.number_input("max_k (elbow cap)", 2, 100, 10, 1, key="opt_cluster_maxk")

with st.sidebar.expander("Explainability (SHAP)", expanded=True):
    shap_enabled = st.checkbox("Enable SHAP", True, key="opt_shap")
    shap_bg_size = st.number_input("Background sample size", 10, 100000, 100, 10, key="opt_shap_bg")
    shap_test_size = st.number_input("Test rows to explain", 10, 100000, 200, 10, key="opt_shap_test")

# with st.sidebar.expander("Numeric normalization (VIF stabilization)", expanded=False):
#     apply_vif_norm = st.checkbox(
#         "Cast numerics to float64 and round to 9 decimals",
#         value=True,
#         #help="Stabilizes VIF across CSV vs Parquet.",
#         key="opt_vifnorm"
#     )

if "cast9_round9" not in st.session_state:
    st.session_state["cast9_round9"] = CAST9_DEFAULT

cast9 = bool(st.session_state["cast9_round9"])

apply_vif_norm = cast9

st.sidebar.subheader("Reproducibility")
seed_global = st.sidebar.number_input(
    "Random seed",
    min_value=0, max_value=2_147_483_647, value=42, step=1,
    help="Controls random split, model refit, stress noise, clustering, and SHAP sampling.",
    key="opt_seed"
)

with st.sidebar.expander("Correlation Settings", expanded=False):
    corr_method = st.radio("Method", ["pearson", "spearman"], index=0, horizontal=True, key="opt_corr_method")
    corr_cap = st.slider("Heatmap features (default 20, max 60)", min_value=10, max_value=60, value=20, step=5, key="opt_corr_cap")
    corr_thr = st.number_input("High-correlation threshold (|r| ≥)", min_value=0.0, max_value=0.99, value=0.80, step=0.05, key="opt_corr_thr")
    st.caption("Tip: ≥ 0.90 often means near-duplicate; confirm with VIF.")

corr_ui_cfg = {
    "enabled": bool(corr_enabled),
    "method": corr_method,
    "high_corr_threshold": float(corr_thr),
    "heatmap_max_features_default": int(corr_cap),
    "heatmap_max_features_limit": 60,
    "subset_strategy": "cluster",
    "top_pairs_max": 200,
    "sample_rows": 150_000,
    "seed": int(seed_global),
    "save_csv": True,
    "save_fig": True,
    "appendix_csv_cap": None,
}

def render_model_form(y_train, seed_global: int):
    """Return (library, algorithm, params, task) using the 20-model registry,
    but never show per-model seed; we inject sidebar seed automatically.
    """
    task_auto = infer_task_from_target(y_train)
    task = st.radio(
        "Task",
        ["classification", "regression"],
        index=0 if task_auto == "classification" else 1,
        horizontal=True,
        key="mdl_task"
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
        if seed_keys:
            st.caption("ℹ️ Model seed is taken from the sidebar **Random seed**.")

        for name, (typ, choices, helptext) in schema.items():
            if name in seed_keys:
                continue

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
# Main layout — single flow
# ==========================

left, right = st.columns([1.35, 1])

with left:
    st.header("1) Choose data source")
    data_source = st.radio(
        "Data source",
        ("Single cleaned file (you split)", "Already split: Train & Test"),
        index=0, horizontal=True
    )

    saved_raw = None
    cleaned_df = train_df = test_df = None

    if data_source.startswith("Single"):
        st.subheader("Upload files")
        cleaned_file = st.file_uploader(
            "Cleaned dataset (required)",
            key="upl_cleaned"
        )
        raw_file = st.file_uploader(
            "Raw dataset (optional)",
            key="upl_raw_single"
        )

        saved_cleaned = _save_upload(cleaned_file, run_dir)
        saved_raw = _save_upload(raw_file, run_dir)

        df_preview = None
        if saved_cleaned:
            st.success(f"Cleaned file saved: `{saved_cleaned}`")
            try:
                df_preview = load_dataframe(saved_cleaned)
                cleaned_df = df_preview
                st.write("Preview (top 10 rows):")
                st.dataframe(df_preview.head(10), width="stretch")
            except Exception as e:
                st.error(f"Could not read cleaned file: {e}")

        st.subheader("Configure data")
        if df_preview is not None:
            target_default = _pick_target(df_preview)
            cols = list(df_preview.columns)
            target = st.selectbox("Target column", options=cols, index=cols.index(target_default) if target_default in cols else 0)
            features = st.multiselect(
                "Features",
                options=[c for c in cols if c != target],
                default=[c for c in cols if c != target],
            )
        else:
            target, features = None, []

        test_size = st.slider("Hold-out test size", 0.1, 0.5, 0.3, 0.05)

    else:
        st.subheader("Upload TRAIN/TEST (cleaned)")
        train_cleaned = st.file_uploader(
            "Train (cleaned) — required",
            key="upl_train"
        )
        test_cleaned = st.file_uploader(
            "Test (cleaned) — required",
            key="upl_test"
        )
        raw_file = st.file_uploader(
            "Raw dataset (optional, global)",
            key="upl_raw_global"
        )

        saved_train = _save_upload(train_cleaned, run_dir)
        saved_test  = _save_upload(test_cleaned, run_dir)
        saved_raw   = _save_upload(raw_file, run_dir)

        df_tr = df_te = None
        if saved_train:
            try:
                df_tr = load_dataframe(saved_train)
                train_df = df_tr
            except Exception as e:
                st.error(f"Could not read train: {e}")
        if saved_test:
            try:
                df_te = load_dataframe(saved_test)
                test_df = df_te
            except Exception as e:
                st.error(f"Could not read test: {e}")

        if df_tr is not None:
            st.write("Train preview (top 10):")
            st.dataframe(df_tr.head(10), width="stretch")
            target_default = _pick_target(df_tr)
            cols = list(df_tr.columns)
            target = st.selectbox("Target column", options=cols, index=cols.index(target_default) if target_default in cols else 0)
            features = st.multiselect(
                "Features",
                options=[c for c in cols if c != target],
                default=[c for c in cols if c != target],
            )
        else:
            target, features = None, []

with right:
    st.header("2) Refit, Validate & Report")
    tvr_render_ready("refit", header_text="Run & Report (last run)")
    tvr_render_extras("refit")

    report_name = st.text_input(
        "Report file name (.docx)",
        value=f"tanml_report_{int(time.time())}.docx"
    )

    # --- Choose model BEFORE running ---
    if data_source.startswith("Single"):
        if 'cleaned_df' in locals() and cleaned_df is not None and target:
            y_for_task = cleaned_df[target]
        else:
            y_for_task = pd.Series([], dtype="float64")
    else:
        if 'train_df' in locals() and train_df is not None and target:
            y_for_task = train_df[target]
        else:
            y_for_task = pd.Series([], dtype="float64")

    library_selected, algo_selected, user_hp, task_selected = render_model_form(y_for_task, seed_global)

    # ---- Conditional thresholds (left panel) ----
    with st.sidebar:
        if task_selected == "classification":
            st.subheader("Thresholds")
            c1, c2, c3 = st.columns(3)
            auc_min = c1.number_input("AUC ≥", 0.0, 1.0, 0.60, 0.01, key="thr_auc")
            f1_min  = c2.number_input("F1 ≥",  0.0, 1.0, 0.60, 0.01, key="thr_f1")
            ks_min  = c3.number_input("KS ≥",  0.0, 1.0, 0.20, 0.01, key="thr_ks")
            st.session_state["__thr_block__"] = {"AUC_min": auc_min, "F1_min": f1_min, "KS_min": ks_min}
        else:
            # Regression (or anything not classification) → hide thresholds entirely.
            auc_min = 0.0
            f1_min  = 0.0
            ks_min  = 0.0
            st.session_state["__thr_block__"] = {"problem_type": "regression"}

    st.session_state["model_selection"] = {
        "library": library_selected,
        "algo": algo_selected,
        "hp": user_hp,
        "task": task_selected,
    }

    if data_source.startswith("Single"):
        ready = bool((locals().get("saved_cleaned") is not None) and target and features)
    else:
        ready = bool((locals().get("saved_train") is not None) and (locals().get("saved_test") is not None) and target and features)
    ready = ready and bool(st.session_state.get("model_selection", {}).get("algo"))

    go = st.button("▶️ Refit & validate", type="primary", disabled=not ready)
    if not ready:
        st.info("Provide data, pick target + features, choose a model, then run.")

    if go:
        try:
            # ---- Build X_train/X_test/y_train/y_test ----
            if data_source.startswith("Single"):
                cleaned_df = cleaned_df if cleaned_df is not None else load_dataframe(saved_cleaned)
                if target not in cleaned_df.columns:
                    st.error(f"Target '{target}' not found in cleaned data."); st.stop()

                safe_features = [c for c in features if c in cleaned_df.columns and c != target]
                if not safe_features:
                    safe_features = [c for c in cleaned_df.columns if c != target]

                X = cleaned_df[safe_features].copy()
                y = cleaned_df[target].copy()

                if apply_vif_norm:
                    cleaned_df = _normalize_vif(cleaned_df)
                    X = _normalize_vif(X)

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=float(test_size), random_state=seed_global, shuffle=True
                )
                df_checks = pd.concat([X_train, y_train], axis=1)
                if apply_vif_norm:
                    df_checks = _normalize_vif(df_checks)

                split_strategy = "random"
                saved_raw_ = saved_raw

            else:
                train_df = train_df if train_df is not None else load_dataframe(saved_train)
                test_df  = test_df  if test_df  is not None else load_dataframe(saved_test)

                if target not in train_df.columns: st.error(f"Target '{target}' not in TRAIN."); st.stop()
                if target not in test_df.columns:  st.error(f"Target '{target}' not in TEST.");  st.stop()

                X_train = train_df[features].copy(); y_train = train_df[target].copy()
                te_sub  = test_df[features + [target]].copy()
                te_aligned, err = _schema_align_or_error(train_df[features + [target]], te_sub)
                if err: st.error(err); st.stop()
                X_test  = te_aligned[features].copy(); y_test = te_aligned[target].copy()

                if apply_vif_norm:
                    train_df = _normalize_vif(train_df); test_df = _normalize_vif(test_df)
                    X_train = _normalize_vif(X_train);   X_test  = _normalize_vif(X_test)

                df_checks = pd.concat([X_train, y_train], axis=1)
                if apply_vif_norm: df_checks = _normalize_vif(df_checks)

                overlap_pct = _row_overlap_pct(
                    pd.concat([X_train, y_train], axis=1),
                    pd.concat([X_test,  y_test],  axis=1),
                    cols=features + [target]
                )
                if overlap_pct > 0:
                    st.warning(f"Potential Train↔Test row overlap: ~{overlap_pct:.2f}% (by row hash).")

                split_strategy = "supplied"
                saved_raw_ = saved_raw

            # ---- Build estimator from selection ----
            sel = st.session_state.get("model_selection") or {}
            library_selected = sel.get("library")
            algo_selected    = sel.get("algo")
            user_hp          = sel.get("hp") or {}
            task_selected    = sel.get("task")

            if not library_selected or not algo_selected:
                st.error("Please choose a library and algorithm before running.")
                st.stop()

            try:
                model = build_estimator(library_selected, algo_selected, user_hp)
            except ImportError:
                st.error(
                    f"Missing dependency for '{library_selected}.{algo_selected}'. "
                    f"Install the library (e.g., 'pip install {library_selected}') and try again."
                )
                st.stop()

            model.fit(X_train, y_train)
            if not hasattr(model, "feature_names_in_"):
                try:
                    model.feature_names_in_ = X_train.columns.to_numpy()
                except Exception:
                    pass

            component_seeds = _derive_component_seeds(
                seed_global,
                split_random=(split_strategy == "random"),
                stress_enabled=stress_enabled,
                cluster_enabled=cluster_enabled,
                shap_enabled=shap_enabled,
            )

            rule_cfg = _build_rule_cfg(
                saved_raw=saved_raw_,
                auc_min=auc_min, f1_min=f1_min, ks_min=ks_min,
                eda_enabled=eda_enabled, eda_max_plots=int(eda_max_plots),
                corr_enabled=corr_enabled, vif_enabled=vif_enabled,
                raw_data_check_enabled=raw_data_check_enabled and bool(saved_raw_),
                model_meta_enabled=model_meta_enabled,
                stress_enabled=stress_enabled,
                stress_epsilon=stress_epsilon,
                stress_perturb_fraction=stress_perturb_fraction,
                cluster_enabled=cluster_enabled,
                cluster_k=int(cluster_k),
                cluster_max_k=int(cluster_max_k),
                shap_enabled=shap_enabled,
                shap_bg_size=int(shap_bg_size),
                shap_test_size=int(shap_test_size),
                artifacts_dir=artifacts_dir,
                split_strategy=split_strategy,
                test_size=float(test_size) if split_strategy == "random" else 0.0,
                seed_global=seed_global,
                component_seeds=component_seeds,
                in_scope_cols=list(X_train.columns) + [target],
            )


            rule_cfg.setdefault("explainability", {}).setdefault("shap", {})["out_dir"] = str(artifacts_dir)

            # 2) Backward-compat for wrappers that look under "SHAPCheck"
            rule_cfg["SHAPCheck"] = {
                "enabled": bool(shap_enabled),
                "background_size": int(shap_bg_size),
                "sample_size": int(shap_test_size),
                "out_dir": str(artifacts_dir),
            }
            # Inject correlation settings
            rule_cfg.setdefault("CorrelationCheck", {}).update({
                "enabled": bool(corr_enabled), **{
                    "method": corr_method,
                    "high_corr_threshold": float(corr_thr),
                    "heatmap_max_features_default": int(corr_cap),
                    "heatmap_max_features_limit": 60,
                    "subset_strategy": "cluster",
                    "top_pairs_max": 200,
                    "sample_rows": 150_000,
                    "seed": int(seed_global),
                    "save_csv": True,
                    "save_fig": True,
                    "appendix_csv_cap": None,
                }
            })
            rule_cfg.setdefault("correlation", {}).update({"enabled": bool(corr_enabled)})

            raw_df_loaded = load_dataframe(saved_raw_) if saved_raw_ else None

            engine = ValidationEngine(
                model, X_train, X_test, y_train, y_test, rule_cfg, df_checks,
                raw_df=raw_df_loaded, ctx=st.session_state
            )

            eff_path = run_dir / "effective_config.yaml"
            with st.status("Refitting & running checks…", expanded=True) as status:
                def cb(msg: str):
                    try: status.write(msg)
                    except Exception: pass

                results = _try_run_engine(engine, progress_cb=cb)

                # Stamp task for downstream UI/history
                results["task_type"] = task_selected
                results.setdefault("summary", {})["task_type"] = task_selected

                rules_meta = {}
                if task_selected == "classification":
                    rules_meta = {"auc_roc_min": auc_min, "f1_min": f1_min, "ks_min": ks_min}

                results.update({
                    "validation_date": pd.Timestamp.now(tz="America/Chicago").strftime("%Y-%m-%d %H:%M:%S %Z"),
                    "model_path": "(refit in UI)",
                    "validated_by": "TanML UI (Refit-only)",
                    "rules": rules_meta,  # conditional now
                    "data_split": "random" if split_strategy == "random" else "supplied_train_test",
                    "reproducibility": {"seed_global": int(seed_global), "component_seeds": component_seeds, "split_strategy": split_strategy},
                    "model_provenance": {
                        "refit_always": True,
                        "library": library_selected,
                        "algorithm": algo_selected,
                        "hyperparameters_used": user_hp,
                        "hyperparameters_source": "user_form"
                    }
                })

                eff_paths = (
                    {"cleaned": str(saved_cleaned), "raw": str(saved_raw_) if saved_raw_ else None}
                    if split_strategy == "random" else
                    {"train_cleaned": str(saved_train), "test_cleaned": str(saved_test), "raw": str(saved_raw_) if saved_raw_ else None}
                )
                effective_cfg = {
                    "scenario": "Refit",
                    "mode": "Refit & Validate",
                    "paths": {**eff_paths, "artifacts_dir": str(artifacts_dir)},
                    "data": {
                        "target": target,
                        "features": list(X_train.columns),
                        "split": "random" if split_strategy == "random" else "supplied",
                        "test_size": float(test_size) if split_strategy == "random" else None,
                        "random_state": seed_global if split_strategy == "random" else None,
                    },
                    "model_refit": {"library": library_selected, "type": algo_selected, "hyperparameters": user_hp},
                    "checks": rule_cfg,
                    "reproducibility": {"seed_global": int(seed_global), "component_seeds": component_seeds},
                }
                try:
                    from ruamel.yaml import YAML
                    YAML().dump(effective_cfg, eff_path.open("w"))
                except Exception:
                    eff_path.write_text(json.dumps(effective_cfg, indent=2))

                report_path = run_dir / report_name
                report_path.parent.mkdir(parents=True, exist_ok=True)

                # ⬇️ choose correct template based on task (classification/regression)
                template_path = _choose_report_template(task_selected)

                # Build the report with the chosen template
                ReportBuilder(
                    results,
                    template_path=str(template_path),
                    output_path=report_path
                ).build()

                status.update(label="Done!", state="complete", expanded=False)

            tvr_store_extras("refit", {
                "results": results,
                "train_rows": len(y_train),
                "test_rows": len(y_test),
                "target": target,
                "n_features": len(X_train.columns),
                "seed_used": seed_global if split_strategy == "random" else "N/A (user-supplied split)",
                "eff_path": str(eff_path),
                "artifacts_dir": str(artifacts_dir),
            })

            tvr_finish(
                "refit",
                report_path=report_path,
                file_name=report_path.name,
                summary=results.get("summary", {}),
                label=f"Refit — {library_selected}.{algo_selected}",
                cfg=effective_cfg,
            )
            tvr_render_ready("refit", header_text="Run & Report (last run)")
            tvr_render_extras("refit")

        except Exception as e:
            st.error(f"Refit/validate failed: {e}")
            st.stop()


st.markdown("---")
