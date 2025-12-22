# tanml/ui/pages/setup.py
"""
Setup page for TanML UI.

Handles data upload, target selection, and initial configuration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st

from tanml.utils.data_loader import load_dataframe
from tanml.ui.helpers import session_dir, save_upload, pick_target


def render(run_dir: Path) -> None:
    """
    Render the Setup page.
    
    Args:
        run_dir: Directory for artifacts
    """
    st.header("📤 Setup & Data Upload")
    
    # Initialize session state
    if "tvr_raw_df" not in st.session_state:
        st.session_state["tvr_raw_df"] = None
    if "tvr_cleaned_df" not in st.session_state:
        st.session_state["tvr_cleaned_df"] = None
    
    # File upload section
    st.subheader("1. Upload Your Data")
    
    uploaded_files = st.file_uploader(
        "Upload data file(s)",
        type=["csv", "xlsx", "xls", "parquet", "dta", "sav", "sas7bdat"],
        accept_multiple_files=True,
        help="Supported formats: CSV, Excel, Parquet, Stata, SPSS, SAS",
        key="setup_file_upload",
    )
    
    if uploaded_files:
        # Process uploaded files
        for upload in uploaded_files:
            with st.spinner(f"Loading {upload.name}..."):
                save_path = save_upload(upload, run_dir / "uploads")
                df = load_dataframe(save_path)
                
                if df is not None and not df.empty:
                    st.success(f"✅ Loaded {upload.name}: {len(df):,} rows × {len(df.columns)} columns")
                    st.session_state["tvr_raw_df"] = df
                    st.session_state["tvr_cleaned_df"] = df.copy()
                else:
                    st.error(f"Failed to load {upload.name}")
    
    # Show loaded data info
    raw_df = st.session_state.get("tvr_raw_df")
    if raw_df is not None:
        st.divider()
        st.subheader("2. Configure Target")
        
        # Target column selection
        cols = list(raw_df.columns)
        default_idx = pick_target(raw_df)
        
        target_col = st.selectbox(
            "Select target column",
            options=cols,
            index=default_idx,
            key="setup_target_col",
        )
        
        st.session_state["tvr_target_col"] = target_col
        
        # Show data preview
        with st.expander("📊 Data Preview", expanded=True):
            st.dataframe(raw_df.head(10), use_container_width=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", f"{len(raw_df):,}")
            with col2:
                st.metric("Columns", len(raw_df.columns))
            with col3:
                missing_pct = raw_df.isnull().mean().mean() * 100
                st.metric("Missing %", f"{missing_pct:.1f}%")
        
        # Target distribution
        with st.expander("🎯 Target Distribution", expanded=False):
            target_series = raw_df[target_col]
            
            if target_series.nunique() <= 10:
                # Categorical/binary target
                st.bar_chart(target_series.value_counts())
            else:
                # Continuous target
                st.line_chart(target_series.describe())
        
        st.divider()
        st.subheader("3. Proceed to Next Step")
        
        st.info(
            "✅ Data loaded and target configured. "
            "Navigate to **Data Profiling** in the sidebar to explore your data, "
            "or go directly to **Preprocessing** to prepare for modeling."
        )
    else:
        st.info("👆 Upload a data file to get started.")


def get_loaded_data() -> tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Get the currently loaded data and target.
    
    Returns:
        Tuple of (DataFrame, target_column_name)
    """
    df = st.session_state.get("tvr_cleaned_df")
    target = st.session_state.get("tvr_target_col")
    return df, target
