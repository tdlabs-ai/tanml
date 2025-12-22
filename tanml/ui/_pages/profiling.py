# tanml/ui/pages/profiling.py
"""
Data profiling page for TanML UI.

Provides detailed exploratory data analysis and data quality assessment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st


def render(run_dir: Path) -> None:
    """
    Render the Data Profiling page.
    
    Args:
        run_dir: Directory for artifacts
    """
    st.header("🔍 Data Profiling")
    
    # Get data from session state
    raw_df = st.session_state.get("tvr_raw_df")
    cleaned_df = st.session_state.get("tvr_cleaned_df")
    
    if raw_df is None:
        st.warning("⚠️ No data loaded. Please go to Setup and upload data first.")
        return
    
    # Tab selection
    tab1, tab2 = st.tabs(["📊 Raw Data Profile", "✨ Cleaned Data Profile"])
    
    with tab1:
        _render_profile(raw_df, "Raw Data")
    
    with tab2:
        if cleaned_df is not None:
            _render_profile(cleaned_df, "Cleaned Data")
        else:
            st.info("Cleaned data will appear after preprocessing.")


def _render_profile(df: pd.DataFrame, title: str) -> None:
    """
    Render detailed data profile.
    
    Args:
        df: DataFrame to profile
        title: Section title
    """
    st.subheader(f"📋 {title} Overview")
    
    # KPI row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📝 Rows", f"{len(df):,}")
    with col2:
        st.metric("📊 Columns", len(df.columns))
    with col3:
        missing_pct = df.isnull().mean().mean() * 100
        st.metric("❓ Missing %", f"{missing_pct:.1f}%")
    with col4:
        dup_count = df.duplicated().sum()
        st.metric("🔄 Duplicates", f"{dup_count:,}")
    
    st.divider()
    
    # Data quality risks
    st.subheader("⚠️ Data Quality Risks")
    
    risks = _identify_risks(df)
    
    if risks:
        for risk in risks:
            severity = risk.get("severity", "warning")
            icon = "🔴" if severity == "high" else "🟡" if severity == "medium" else "🟢"
            st.warning(f"{icon} **{risk['title']}**: {risk['description']}")
    else:
        st.success("✅ No major data quality issues detected!")
    
    st.divider()
    
    # Detailed tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔢 Numeric", "📝 Categorical", "❓ Missing", "📊 Stats"])
    
    with tab1:
        _render_numeric_profile(df)
    
    with tab2:
        _render_categorical_profile(df)
    
    with tab3:
        _render_missing_profile(df)
    
    with tab4:
        _render_statistics(df)


def _identify_risks(df: pd.DataFrame) -> list[dict]:
    """Identify data quality risks."""
    risks = []
    
    # High missing rate
    missing_rates = df.isnull().mean()
    high_missing = missing_rates[missing_rates > 0.3]
    if not high_missing.empty:
        risks.append({
            "title": "High Missing Rate",
            "description": f"{len(high_missing)} columns have >30% missing values: {', '.join(high_missing.index[:3])}",
            "severity": "high",
        })
    
    # Constant columns
    constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
    if constant_cols:
        risks.append({
            "title": "Constant Columns",
            "description": f"{len(constant_cols)} columns have only one unique value: {', '.join(constant_cols[:3])}",
            "severity": "medium",
        })
    
    # High cardinality
    high_card = []
    for col in df.select_dtypes(include=["object", "category"]).columns:
        if df[col].nunique() > 100:
            high_card.append(col)
    if high_card:
        risks.append({
            "title": "High Cardinality",
            "description": f"{len(high_card)} categorical columns have >100 unique values: {', '.join(high_card[:3])}",
            "severity": "medium",
        })
    
    # Duplicates
    dup_pct = df.duplicated().mean() * 100
    if dup_pct > 5:
        risks.append({
            "title": "Duplicated Rows",
            "description": f"{dup_pct:.1f}% of rows are duplicates",
            "severity": "medium" if dup_pct < 20 else "high",
        })
    
    return risks


def _render_numeric_profile(df: pd.DataFrame) -> None:
    """Render numeric columns profile."""
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    
    if not numeric_cols:
        st.info("No numeric columns found.")
        return
    
    st.write(f"**{len(numeric_cols)} Numeric Columns**")
    
    # Summary statistics
    stats_df = df[numeric_cols].describe().T
    stats_df["missing_%"] = df[numeric_cols].isnull().mean() * 100
    stats_df["zeros_%"] = (df[numeric_cols] == 0).mean() * 100
    
    st.dataframe(
        stats_df[["count", "mean", "std", "min", "max", "missing_%", "zeros_%"]].round(2),
        use_container_width=True,
    )


def _render_categorical_profile(df: pd.DataFrame) -> None:
    """Render categorical columns profile."""
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    
    if not cat_cols:
        st.info("No categorical columns found.")
        return
    
    st.write(f"**{len(cat_cols)} Categorical Columns**")
    
    # Summary
    cat_summary = []
    for col in cat_cols:
        cat_summary.append({
            "Column": col,
            "Unique": df[col].nunique(),
            "Missing %": f"{df[col].isnull().mean() * 100:.1f}%",
            "Top Value": str(df[col].mode().iloc[0]) if not df[col].mode().empty else "N/A",
            "Top Count": df[col].value_counts().iloc[0] if not df[col].value_counts().empty else 0,
        })
    
    st.dataframe(pd.DataFrame(cat_summary), use_container_width=True)


def _render_missing_profile(df: pd.DataFrame) -> None:
    """Render missing values profile."""
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    
    if missing.empty:
        st.success("✅ No missing values!")
        return
    
    st.write(f"**{len(missing)} Columns with Missing Values**")
    
    missing_df = pd.DataFrame({
        "Column": missing.index,
        "Missing Count": missing.values,
        "Missing %": (missing.values / len(df) * 100).round(2),
    })
    
    st.dataframe(missing_df, use_container_width=True)
    
    # Bar chart
    st.bar_chart(missing_df.set_index("Column")["Missing %"].head(20))


def _render_statistics(df: pd.DataFrame) -> None:
    """Render full statistics."""
    st.write("**Full Dataset Statistics**")
    st.dataframe(df.describe(include="all").T, use_container_width=True)
