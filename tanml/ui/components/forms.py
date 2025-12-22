# tanml/ui/components/forms.py
"""
Form components for TanML UI.

Provides model configuration forms and file uploaders.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

from tanml.models.registry import ui_schema_for


def render_model_form(
    model_name: str,
    key_prefix: str = "model_form",
) -> Dict[str, Any]:
    """
    Render a dynamic model configuration form.
    
    Uses the model's UI schema from the registry to generate
    appropriate input widgets for each parameter.
    
    Args:
        model_name: Name of the model (from registry)
        key_prefix: Prefix for Streamlit widget keys
        
    Returns:
        Dictionary of parameter name -> value
        
    Example:
        params = render_model_form("RandomForestClassifier")
        # Returns: {"n_estimators": 100, "max_depth": 10, ...}
    """
    schema = ui_schema_for(model_name)
    if not schema:
        st.info("No configurable parameters for this model.")
        return {}
    
    params = {}
    
    for param_schema in schema:
        name = param_schema.get("name")
        param_type = param_schema.get("type", "text")
        default = param_schema.get("default")
        label = param_schema.get("label", name)
        help_text = param_schema.get("help", "")
        
        key = f"{key_prefix}_{name}"
        
        if param_type == "int":
            min_val = param_schema.get("min", 1)
            max_val = param_schema.get("max", 1000)
            params[name] = st.number_input(
                label,
                min_value=min_val,
                max_value=max_val,
                value=default or min_val,
                step=1,
                key=key,
                help=help_text,
            )
        
        elif param_type == "float":
            min_val = param_schema.get("min", 0.0)
            max_val = param_schema.get("max", 1.0)
            step = param_schema.get("step", 0.01)
            params[name] = st.number_input(
                label,
                min_value=min_val,
                max_value=max_val,
                value=default or min_val,
                step=step,
                key=key,
                help=help_text,
            )
        
        elif param_type == "select":
            options = param_schema.get("options", [])
            default_idx = options.index(default) if default in options else 0
            params[name] = st.selectbox(
                label,
                options=options,
                index=default_idx,
                key=key,
                help=help_text,
            )
        
        elif param_type == "bool":
            params[name] = st.checkbox(
                label,
                value=bool(default),
                key=key,
                help=help_text,
            )
        
        elif param_type == "multiselect":
            options = param_schema.get("options", [])
            params[name] = st.multiselect(
                label,
                options=options,
                default=default or [],
                key=key,
                help=help_text,
            )
        
        else:  # text/other
            params[name] = st.text_input(
                label,
                value=str(default) if default else "",
                key=key,
                help=help_text,
            )
    
    return params


def file_uploader_multi(
    label: str = "Upload data files",
    accepted_types: Optional[List[str]] = None,
    key: str = "file_upload",
    help_text: Optional[str] = None,
) -> List[Any]:
    """
    Multi-file uploader with support for various data formats.
    
    Args:
        label: Upload label
        accepted_types: List of accepted extensions (default: common data formats)
        key: Streamlit widget key
        help_text: Optional help text
        
    Returns:
        List of uploaded file objects
    """
    if accepted_types is None:
        accepted_types = [
            "csv", "xlsx", "xls",  # Tabular
            "parquet",            # Columnar
            "dta", "sav", "sas7bdat",  # Statistical
            "json",               # JSON
        ]
    
    default_help = (
        "Supported formats: CSV, Excel, Parquet, Stata (.dta), "
        "SPSS (.sav), SAS (.sas7bdat)"
    )
    
    files = st.file_uploader(
        label,
        type=accepted_types,
        accept_multiple_files=True,
        key=key,
        help=help_text or default_help,
    )
    
    return files or []


def target_column_selector(
    columns: List[str],
    key: str = "target_col",
    label: str = "Target Column",
) -> Optional[str]:
    """
    Select target column from available columns.
    
    Args:
        columns: List of column names
        key: Streamlit widget key
        label: Widget label
        
    Returns:
        Selected column name or None
    """
    if not columns:
        st.warning("No columns available for selection.")
        return None
    
    # Try to auto-detect common target names
    common_targets = ["target", "label", "y", "class", "outcome"]
    default_idx = 0
    
    for i, col in enumerate(columns):
        if col.lower() in common_targets:
            default_idx = i
            break
    
    return st.selectbox(
        label,
        options=columns,
        index=default_idx,
        key=key,
    )


def split_ratio_slider(
    key: str = "split_ratio",
    default: float = 0.2,
    min_val: float = 0.1,
    max_val: float = 0.5,
) -> float:
    """
    Slider for train/test split ratio.
    
    Args:
        key: Streamlit widget key
        default: Default split ratio
        min_val: Minimum ratio
        max_val: Maximum ratio
        
    Returns:
        Selected split ratio
    """
    return st.slider(
        "Test Split Ratio",
        min_value=min_val,
        max_value=max_val,
        value=default,
        step=0.05,
        key=key,
        help="Proportion of data to use for testing",
    )
