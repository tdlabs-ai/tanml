# tanml/ui/services/data.py
"""
Data loading and validation services for TanML UI.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import streamlit as st

from tanml.utils.data_loader import load_dataframe


def _save_upload(upload, dest_dir: Path, custom_name: str | None = None) -> Path | None:
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


def _schema_align_or_error(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[pd.DataFrame, str | None]:
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
        if pd.api.types.is_numeric_dtype(td) and not pd.api.types.is_numeric_dtype(
            aligned[c].dtype
        ):
            aligned[c] = pd.to_numeric(aligned[c], errors="coerce")
    return aligned, None


def _row_overlap_pct(train_df: pd.DataFrame, test_df: pd.DataFrame, cols: list[str]) -> float:
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
