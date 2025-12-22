# tanml/ui/services/data.py
"""
Data handling services for TanML UI.

Provides utilities for loading, saving, and preprocessing data files.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd
import streamlit as st

from tanml.utils.data_loader import load_dataframe


def save_uploaded_file(
    uploaded_file: Any,
    save_dir: Union[str, Path],
    preserve_extension: bool = True,
) -> Path:
    """
    Save an uploaded file to disk.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        save_dir: Directory to save to
        preserve_extension: Whether to keep original extension
        
    Returns:
        Path to the saved file
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Get original filename and extension
    original_name = getattr(uploaded_file, "name", "data.csv")
    
    if preserve_extension:
        # Keep original extension
        save_path = save_dir / original_name
    else:
        # Default to CSV
        save_path = save_dir / f"{Path(original_name).stem}.csv"
    
    # Write file
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return save_path


def load_data_file(
    path: Union[str, Path],
    **kwargs,
) -> pd.DataFrame:
    """
    Load a data file into a DataFrame.
    
    Automatically detects file format from extension.
    
    Args:
        path: Path to the data file
        **kwargs: Additional arguments for pandas read function
        
    Returns:
        Loaded DataFrame
        
    Raises:
        ValueError: If file format is not supported
    """
    return load_dataframe(path, **kwargs)


def get_data_hash(df: pd.DataFrame) -> str:
    """
    Get a hash of a DataFrame for caching.
    
    Args:
        df: DataFrame to hash
        
    Returns:
        MD5 hash string
    """
    return hashlib.md5(
        pd.util.hash_pandas_object(df).values.tobytes()
    ).hexdigest()[:8]


def validate_data(df: pd.DataFrame, target_col: str) -> tuple[bool, list[str]]:
    """
    Validate data before model training.
    
    Args:
        df: DataFrame to validate
        target_col: Target column name
        
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    
    # Check target exists
    if target_col not in df.columns:
        issues.append(f"Target column '{target_col}' not found in data")
    
    # Check for empty data
    if df.empty:
        issues.append("DataFrame is empty")
    
    # Check for all-null columns
    null_cols = df.columns[df.isnull().all()].tolist()
    if null_cols:
        issues.append(f"All-null columns: {', '.join(null_cols[:5])}")
    
    # Check target has variance
    if target_col in df.columns:
        if df[target_col].nunique() < 2:
            issues.append(f"Target column has less than 2 unique values")
    
    # Check for reasonable row count
    if len(df) < 10:
        issues.append(f"Very few rows ({len(df)}), may not be enough for training")
    
    return len(issues) == 0, issues


def get_feature_columns(df: pd.DataFrame, target_col: str) -> list[str]:
    """
    Get feature columns (all columns except target).
    
    Args:
        df: DataFrame
        target_col: Target column name
        
    Returns:
        List of feature column names
    """
    return [col for col in df.columns if col != target_col]


def infer_column_types(df: pd.DataFrame) -> dict[str, str]:
    """
    Infer column types for display.
    
    Args:
        df: DataFrame
        
    Returns:
        Dictionary of column -> type string
    """
    type_map = {}
    
    for col in df.columns:
        dtype = df[col].dtype
        
        if pd.api.types.is_numeric_dtype(dtype):
            if pd.api.types.is_integer_dtype(dtype):
                type_map[col] = "integer"
            else:
                type_map[col] = "float"
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            type_map[col] = "datetime"
        elif pd.api.types.is_categorical_dtype(dtype):
            type_map[col] = "categorical"
        elif pd.api.types.is_bool_dtype(dtype):
            type_map[col] = "boolean"
        else:
            type_map[col] = "string"
    
    return type_map
