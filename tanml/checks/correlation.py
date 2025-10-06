# checks/correlation.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import squareform
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

from .base import BaseCheck


DEFAULT_CFG = {
    "method": "pearson",                 # "pearson" | "spearman"
    "high_corr_threshold": 0.80,         # |r| >= threshold flagged
    "top_pairs_max": 200,                # rows in the "main" table CSV
    "heatmap_max_features_default": 20,  # default plotted features
    "heatmap_max_features_limit": 60,    # max allowed via UI/slider
    "subset_strategy": "cluster",        # "cluster" | "degree"
    "sample_rows": 150_000,              # downsample for speed on huge data
    "seed": 42,
    "save_csv": True,
    "save_fig": True,
    "appendix_csv_cap": None,            # None = no cap; or int (e.g., 5000)
}


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _numeric_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def _drop_constant_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    keep, dropped = [], []
    for c in df.columns:
        s = df[c]
        if s.dropna().nunique() <= 1: 
            dropped.append(c)
        else:
            keep.append(c)
    return df[keep], dropped


def _subset_by_degree(corr_abs: pd.DataFrame, max_feats: int) -> List[str]:
    if corr_abs.shape[0] <= max_feats:
        return list(corr_abs.index)
    scores = corr_abs.sum().sort_values(ascending=False)
    return list(scores.head(max_feats).index)


def _subset_by_cluster(corr_abs: pd.DataFrame, max_feats: int) -> List[str]:
    if corr_abs.shape[0] <= max_feats:
        return list(corr_abs.index)
    if not _HAS_SCIPY:
        return _subset_by_degree(corr_abs, max_feats)
    # distance = 1 - |corr|
    dist = 1.0 - corr_abs
    dist = (dist + dist.T) / 2.0
    np.fill_diagonal(dist.values, 0.0)
    Z = linkage(squareform(dist.values, checks=False), method="average")
    order = leaves_list(Z)
    ordered = corr_abs.index[order]
    step = max(1, len(ordered) // max_feats)
    return list(ordered[::step][:max_feats])


def _render_heatmap(corr: pd.DataFrame, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr.values, vmin=-1, vmax=1)
    ax.set_xticks(range(corr.shape[1]))
    ax.set_yticks(range(corr.shape[0]))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=6)
    ax.set_yticklabels(corr.index, fontsize=6)
    ax.set_title(title)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Correlation")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


class CorrelationCheck(BaseCheck):
    """
    Numeric-only correlation analysis:
      • Pearson or Spearman (pairwise complete obs)
      • Heatmap on ≤20 features by default (clustered subset up to 60 max)
      • CSV of high-correlation pairs (|r| ≥ threshold), sorted by |r|
      • Handles constant/all-NA columns, optional sampling for speed
    """

    def __init__(
        self,
        cleaned_data: pd.DataFrame,
        cfg: Dict | None = None,
        output_dir: str = "reports/correlation",
    ):
        self.cleaned_data = cleaned_data
        self.cfg = {**DEFAULT_CFG, **(cfg or {})}
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def _top_corr_pairs(self, corr: pd.DataFrame, thr: float) -> pd.DataFrame:
        a = corr.copy()
        np.fill_diagonal(a.values, np.nan)
        s = a.stack().reset_index()
        s.columns = ["feature_i", "feature_j", "corr"]
        s = s.dropna()
        # remove duplicate symmetric pairs
        s["pair"] = s.apply(lambda r: tuple(sorted([r["feature_i"], r["feature_j"]])), axis=1)
        s = s.drop_duplicates(subset=["pair"]).drop(columns=["pair"])
        s["abs_corr"] = s["corr"].abs()
        s = s[s["abs_corr"] >= thr].sort_values("abs_corr", ascending=False)
        return s

    def run(self):
        cfg = self.cfg
        method = cfg["method"]
        thr = float(cfg["high_corr_threshold"])
        top_cap = int(cfg["top_pairs_max"])
        default_cap = int(cfg["heatmap_max_features_default"])
        max_cap = int(cfg["heatmap_max_features_limit"])
        subset_strategy = cfg["subset_strategy"]
        sample_rows = int(cfg["sample_rows"])
        seed = int(cfg["seed"])
        save_csv = bool(cfg["save_csv"])
        save_fig = bool(cfg["save_fig"])
        appendix_cap = cfg.get("appendix_csv_cap", None)

        # ===== 1) Select numeric & (optional) sample rows =====
        X = self.cleaned_data.copy()
        if len(X) > sample_rows:
            X = X.sample(sample_rows, random_state=seed)

        num_cols = _numeric_columns(X)
        X = X[num_cols]
        X, dropped_constants = _drop_constant_columns(X)

        if X.shape[1] < 2:
            msg = "⚠️ Not enough numeric features for correlation."
            print(msg)
            return {
                "pearson_csv": None,
                "spearman_csv": None,
                "heatmap_path": None,
                "top_pairs_csv": None,
                "summary": {"n_numeric_features": X.shape[1]},
                "notes": [msg, f"Dropped constant/all-NA columns: {dropped_constants}"] if dropped_constants else [msg],
                "error": "Not enough numeric features for correlation",
            }

        # ===== 2) Correlation matrix =====
        # Compute both; pick one to drive plotting/threshold logic
        corr_pearson = X.corr(method="pearson")
        corr_spearman = X.corr(method="spearman")
        corr = corr_pearson if method == "pearson" else corr_spearman
        corr_abs = corr.abs()

        # ===== 3) High-correlation pairs CSV =====
        pairs = self._top_corr_pairs(corr, thr)
        # augment with pairwise n_used and feature missingness %
        non_null_counts = X.notna().sum()
        total_rows = len(X)
        if not pairs.empty:
            pairs["n_used"] = pairs.apply(
                lambda r: X[[r["feature_i"], r["feature_j"]]].dropna().shape[0], axis=1
            )
            pairs["pct_missing_i"] = pairs.apply(
                lambda r: 1 - non_null_counts[r["feature_i"]] / total_rows, axis=1
            )
            pairs["pct_missing_j"] = pairs.apply(
                lambda r: 1 - non_null_counts[r["feature_j"]] / total_rows, axis=1
            )

        artifacts: Dict[str, str] = {}
        outdir = Path(self.output_dir)
        _ensure_dir(outdir)

        # Save full correlation matrices (if enabled)
        pearson_csv_path = outdir / "pearson_corr.csv"
        spearman_csv_path = outdir / "spearman_corr.csv"
        if save_csv:
            corr_pearson.to_csv(pearson_csv_path, index=True)
            corr_spearman.to_csv(spearman_csv_path, index=True)

        # Save top-pairs CSVs (main + full/appendix)
        if save_csv:
            full_csv = outdir / "correlation_top_pairs.csv"
            if appendix_cap is not None:
                pairs.head(int(appendix_cap)).to_csv(full_csv, index=False)
            else:
                pairs.to_csv(full_csv, index=False)
            artifacts["top_pairs_csv"] = str(full_csv)

            main_csv = outdir / "correlation_top_pairs_main.csv"
            pairs.head(top_cap).to_csv(main_csv, index=False)
            artifacts["top_pairs_main_csv"] = str(main_csv)

        # ===== 4) Adaptive heatmap =====
        n_features_total = X.shape[1]
        plotted_full_matrix = n_features_total <= default_cap

        if not plotted_full_matrix:
            cap = min(max_cap, n_features_total)
            if subset_strategy == "cluster" and _HAS_SCIPY:
                subset = _subset_by_cluster(corr_abs, cap)
            else:
                subset = _subset_by_degree(corr_abs, cap)
            corr_plot = corr.loc[subset, subset]
            title = f"Correlation Heatmap ({method}) — {len(subset)}/{n_features_total} features (subset)"
        else:
            corr_plot = corr
            title = f"Correlation Heatmap ({method}) — full matrix ({n_features_total} features)"

        heatmap_path = None
        if save_fig:
            heatmap_path = outdir / "heatmap.png"
            _render_heatmap(corr_plot, heatmap_path, title)
            artifacts["heatmap_path"] = str(heatmap_path)

        # ===== 5) Summary/notes =====
        n_pairs_total = n_features_total * (n_features_total - 1) // 2
        n_pairs_flagged = int(pairs.shape[0]) if not pairs.empty else 0
        notes = []
        if dropped_constants:
            notes.append(f"Dropped constant/all-NA columns: {sorted(dropped_constants)}")
        if len(self.cleaned_data) > sample_rows:
            notes.append(f"Computed on a {sample_rows}-row sample (seed={seed}).")
        if not plotted_full_matrix:
            notes.append(
                f"Heatmap shows a subset ({corr_plot.shape[0]}/{n_features_total}); see CSV for full list of pairs."
            )

        return {
            "pearson_csv": str(pearson_csv_path) if save_csv else None,
            "spearman_csv": str(spearman_csv_path) if save_csv else None,
            "heatmap_path": str(heatmap_path) if heatmap_path else None,
            "top_pairs_csv": artifacts.get("top_pairs_csv"),
            "top_pairs_main_csv": artifacts.get("top_pairs_main_csv"),
            "summary": {
                "n_numeric_features": int(n_features_total),
                "n_pairs_total": int(n_pairs_total),
                "n_pairs_flagged_ge_threshold": int(n_pairs_flagged),
                "threshold": float(thr),
                "method": method,
                "plotted_features": int(corr_plot.shape[0]),
                "plotted_full_matrix": bool(plotted_full_matrix),
            },
            "notes": notes,
        }
