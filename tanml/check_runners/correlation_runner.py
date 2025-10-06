# tanml/check_runners/correlation_runner.py
from __future__ import annotations
import os
from typing import Any, Dict, List
import pandas as pd

from tanml.checks.correlation import CorrelationCheck  

def _resolve_outdir(config: Dict[str, Any]) -> str:
    base = (config.get("options") or {}).get("save_artifacts_dir") or "reports"
    outdir = os.path.join(base, "correlation")
    os.makedirs(outdir, exist_ok=True)
    return outdir

def _df_features_only(cleaned_df: pd.DataFrame) -> pd.DataFrame:
    if cleaned_df is None or cleaned_df.empty:
        return cleaned_df
    cols = list(cleaned_df.columns)
    if len(cols) >= 2:
        return cleaned_df[cols[:-1]]
    return cleaned_df

def CorrelationCheckRunner(
    model,
    X_train,
    X_test,
    y_train,
    y_test,
    config: Dict[str, Any],
    cleaned_df: pd.DataFrame,
    raw_df: pd.DataFrame | None = None,
):
    ui_block: Dict[str, Any] = (config.get("CorrelationCheck") or {})
    legacy: Dict[str, Any] = (config.get("correlation") or {})
    if not bool(ui_block.get("enabled", legacy.get("enabled", True))):
        return None

    df = _df_features_only(cleaned_df)
    cfg: Dict[str, Any] = {
        "method": ui_block.get("method", "pearson"),
        "high_corr_threshold": float(ui_block.get("high_corr_threshold", 0.8)),
        "heatmap_max_features_default": int(ui_block.get("heatmap_max_features_default", 20)),
        "heatmap_max_features_limit": int(ui_block.get("heatmap_max_features_limit", 60)),
        "subset_strategy": ui_block.get("subset_strategy", "cluster"),
        "sample_rows": int(ui_block.get("sample_rows", 150_000)),
        "seed": int(ui_block.get("seed", 42)),
        "save_csv": True,
        "save_fig": True,
        "appendix_csv_cap": ui_block.get("appendix_csv_cap", None),
    }
    outdir = _resolve_outdir(config)
    return CorrelationCheck(cleaned_data=df, cfg=cfg, output_dir=outdir).run()
