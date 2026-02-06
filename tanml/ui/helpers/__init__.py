# tanml/ui/helpers.py
"""
General helper functions for TanML UI.

Small utility functions used across the UI.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import streamlit as st


def session_dir() -> Path:
    """
    Get or create the session directory for artifacts.

    Returns:
        Path to session-specific directory
    """
    if "run_dir" not in st.session_state:
        base = Path(tempfile.gettempdir()) / "tanml_runs"
        date_str = datetime.now().strftime("%Y%m%d")
        session_id = st.session_state.get("session_id", os.urandom(4).hex())
        st.session_state["session_id"] = session_id

        run_dir = base / date_str / session_id
        run_dir.mkdir(parents=True, exist_ok=True)
        st.session_state["run_dir"] = run_dir

    return st.session_state["run_dir"]


def save_upload(
    upload,
    dest_dir: Path,
    custom_name: str | None = None,
) -> Path:
    """
    Save an uploaded file to disk, preserving extension.

    Args:
        upload: Streamlit UploadedFile object
        dest_dir: Destination directory
        custom_name: Optional custom filename (without extension)

    Returns:
        Path to saved file
    """
    dest_dir.mkdir(parents=True, exist_ok=True)

    original_name = getattr(upload, "name", "data.csv")
    ext = Path(original_name).suffix or ".csv"

    if custom_name:
        filename = f"{custom_name}{ext}"
    else:
        filename = original_name

    save_path = dest_dir / filename

    with open(save_path, "wb") as f:
        f.write(upload.getbuffer())

    return save_path


def pick_target(df: pd.DataFrame) -> int:
    """
    Pick default target column index.

    Prefers columns named 'target', 'label', 'y', otherwise last column.

    Args:
        df: DataFrame

    Returns:
        Index of target column
    """
    cols_lower = [c.lower() for c in df.columns]
    for name in ["target", "label", "y", "class", "outcome"]:
        if name in cols_lower:
            return cols_lower.index(name)
    return len(df.columns) - 1


def normalize_vif(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize numeric columns for stable VIF calculation.

    Casts numerics to float64 and rounds to 9 decimals.

    Args:
        df: DataFrame to normalize

    Returns:
        Normalized DataFrame (copy)
    """
    df = df.copy()
    for col in df.select_dtypes(include=["number"]).columns:
        df[col] = df[col].astype("float64").round(9)
    return df


def schema_align_or_error(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, str | None]:
    """
    Align test DataFrame schema to training DataFrame.

    - Drops extra columns in test
    - Returns error if columns missing in test
    - Coerces dtypes when safe

    Args:
        train_df: Training DataFrame (reference schema)
        test_df: Test DataFrame to align

    Returns:
        Tuple of (aligned_test_df, error_message or None)
    """
    train_cols = set(train_df.columns)
    test_cols = set(test_df.columns)

    # Find missing columns
    missing = train_cols - test_cols
    if missing:
        return test_df, f"Missing columns in test data: {', '.join(sorted(missing)[:5])}"

    # Drop extra columns
    extra = test_cols - train_cols
    if extra:
        test_df = test_df.drop(columns=list(extra))

    # Reorder to match training
    test_df = test_df[train_df.columns.tolist()]

    # Coerce dtypes
    for col in train_df.columns:
        train_dtype = train_df[col].dtype
        test_dtype = test_df[col].dtype

        if train_dtype != test_dtype:
            # Try to coerce numeric to numeric
            if pd.api.types.is_numeric_dtype(train_dtype) and pd.api.types.is_numeric_dtype(
                test_dtype
            ):
                test_df[col] = test_df[col].astype(train_dtype)

    return test_df, None


def row_overlap_pct(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cols: list[str],
) -> float:
    """
    Calculate row overlap percentage (leakage check).

    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame
        cols: Columns to use for comparison

    Returns:
        Percentage of test rows that appear in training (0-100)
    """

    def hash_rows(df: pd.DataFrame) -> set:
        hashes = set()
        for _, row in df[cols].iterrows():
            row_str = ",".join(str(v) for v in row.values)
            h = hashlib.md5(row_str.encode()).hexdigest()
            hashes.add(h)
        return hashes

    if not cols or train_df.empty or test_df.empty:
        return 0.0

    train_hashes = hash_rows(train_df)
    test_hashes = hash_rows(test_df)

    overlap = len(train_hashes & test_hashes)
    return 100.0 * overlap / len(test_hashes) if test_hashes else 0.0


def derive_component_seeds(
    global_seed: int,
    split_random: bool = False,
    stress_enabled: bool = False,
    cluster_enabled: bool = False,
    shap_enabled: bool = False,
) -> dict[str, int | None]:
    """
    Derive component-specific seeds from global seed.

    Args:
        global_seed: Master random seed
        split_random: Whether to randomize split
        stress_enabled: Whether stress test is enabled
        cluster_enabled: Whether cluster check is enabled
        shap_enabled: Whether SHAP is enabled

    Returns:
        Dictionary of component -> seed
    """
    return {
        "split": None if split_random else global_seed,
        "stress": global_seed + 1 if stress_enabled else None,
        "cluster": global_seed + 2 if cluster_enabled else None,
        "shap": global_seed + 3 if shap_enabled else None,
    }


def fmt2(v, decimals: int = 2, dash: str = "—") -> str:
    """
    Format a numeric value with specified decimals.

    Args:
        v: Value to format
        decimals: Number of decimal places
        dash: String to return for None/NaN

    Returns:
        Formatted string
    """
    if v is None:
        return dash
    try:
        vf = float(v)
        if np.isnan(vf):
            return dash
        return f"{vf:.{decimals}f}"
    except (ValueError, TypeError):
        return dash


def get_value_or_default(d: dict, *keys, default=None):
    """
    Get first available value from dict using multiple keys.

    Args:
        d: Dictionary to search
        *keys: Keys to try in order
        default: Default if no key found

    Returns:
        First found value or default
    """
    for key in keys:
        if key in d and d[key] is not None:
            return d[key]
    return default


# Legacy aliases
_session_dir = session_dir
_save_upload = save_upload
_pick_target = pick_target
_normalize_vif = normalize_vif
_schema_align_or_error = schema_align_or_error
_row_overlap_pct = row_overlap_pct
_derive_component_seeds = derive_component_seeds
_fmt2 = fmt2
_g = get_value_or_default
