# tanml/ui/components/renderers.py
"""
Specialized rendering components for TanML UI.

These components handle displaying validation results, correlation outputs,
and regression outputs.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from tanml.ui.helpers import fmt2


def render_correlation_outputs(
    results: Dict[str, Any],
    title: str = "Numeric Correlation",
) -> None:
    """
    Render correlation check outputs.
    
    Args:
        results: Correlation check results dictionary
        title: Section title
    """
    st.subheader(f"📊 {title}")
    
    # Extract correlation data
    corr_check = results.get("CorrelationCheck", {})
    if isinstance(corr_check, dict) and "CorrelationCheck" in corr_check:
        corr_check = corr_check["CorrelationCheck"]
    
    if not corr_check:
        st.info("Correlation check was not run or produced no results.")
        return
    
    # Summary metrics
    summary = corr_check.get("summary", {})
    artifacts = corr_check.get("artifacts", corr_check)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Method", summary.get("method", "Pearson").title())
    with col2:
        st.metric("Threshold", fmt2(summary.get("threshold", 0.8)))
    with col3:
        st.metric("Features", fmt2(summary.get("n_numeric_features")))
    with col4:
        st.metric("High Pairs", fmt2(summary.get("n_pairs_flagged_ge_threshold", 0)))
    
    # Heatmap
    heatmap_path = artifacts.get("heatmap_path") or artifacts.get("heatmap")
    if heatmap_path and os.path.exists(heatmap_path):
        st.image(heatmap_path, caption="Correlation Heatmap", use_container_width=True)
    
    # Top pairs table
    top_pairs = corr_check.get("top_pairs", [])
    if top_pairs:
        st.write("**Top Correlated Pairs**")
        pairs_df = pd.DataFrame(top_pairs)
        st.dataframe(pairs_df.head(20), use_container_width=True)
    
    # CSV download
    csv_path = artifacts.get("top_pairs_csv") or artifacts.get("csv")
    if csv_path and os.path.exists(csv_path):
        with open(csv_path, "r") as f:
            csv_content = f.read()
        st.download_button(
            "📥 Download Correlation CSV",
            csv_content,
            file_name="correlations.csv",
            mime="text/csv",
        )


def render_regression_outputs(results: Dict[str, Any]) -> None:
    """
    Render regression metrics and diagnostics.
    
    Args:
        results: Results dictionary containing RegressionMetrics
    """
    st.subheader("📈 Regression Metrics")
    
    reg_metrics = results.get("RegressionMetrics", {})
    if isinstance(reg_metrics, dict) and "RegressionMetrics" in reg_metrics:
        reg_metrics = reg_metrics["RegressionMetrics"]
    
    if not reg_metrics:
        st.info("Regression metrics not available.")
        return
    
    # Main metrics tiles
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        rmse = reg_metrics.get("rmse")
        st.metric("RMSE", fmt2(rmse, 4))
    with col2:
        mae = reg_metrics.get("mae")
        st.metric("MAE", fmt2(mae, 4))
    with col3:
        r2 = reg_metrics.get("r2")
        st.metric("R²", fmt2(r2, 4))
    with col4:
        median_ae = reg_metrics.get("median_ae")
        st.metric("Median AE", fmt2(median_ae, 4))
    
    # Additional metrics
    with st.expander("📊 Detailed Metrics", expanded=False):
        r2_adj = reg_metrics.get("r2_adjusted")
        mape_label = reg_metrics.get("mape_or_smape")
        mape_used = reg_metrics.get("mape_used")
        
        if r2_adj is not None:
            st.write(f"**Adjusted R²**: {fmt2(r2_adj, 4)}")
        if mape_label is not None:
            label = "MAPE" if mape_used else "SMAPE"
            st.write(f"**{label}**: {fmt2(mape_label, 2)}%")
        
        notes = reg_metrics.get("notes", [])
        if notes:
            st.write("**Notes:**")
            for note in notes:
                st.write(f"- {note}")
    
    # Diagnostic plots
    artifacts = reg_metrics.get("artifacts", {})
    
    plot_cols = st.columns(2)
    
    plots = [
        ("pred_vs_actual", "Predicted vs Actual"),
        ("residuals_vs_pred", "Residuals vs Predicted"),
        ("residual_hist", "Residual Distribution"),
        ("qq_plot", "Q-Q Plot"),
    ]
    
    for i, (key, label) in enumerate(plots):
        plot_path = artifacts.get(key)
        if plot_path and os.path.exists(plot_path):
            with plot_cols[i % 2]:
                st.image(plot_path, caption=label, use_container_width=True)


def render_classification_outputs(results: Dict[str, Any]) -> None:
    """
    Render classification metrics and diagnostics.
    
    Args:
        results: Results dictionary containing classification summary
    """
    st.subheader("🎯 Classification Metrics")
    
    cls_summary = results.get("classification_summary", {})
    
    if not cls_summary:
        # Try alternative locations
        perf = results.get("performance", {}).get("classification", {})
        cls_summary = perf.get("summary", {})
    
    if not cls_summary:
        st.info("Classification metrics not available.")
        return
    
    # Main metrics tiles
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        auc = cls_summary.get("AUC") or cls_summary.get("auc") or cls_summary.get("roc_auc")
        st.metric("ROC AUC", fmt2(auc, 4))
    with col2:
        ks = cls_summary.get("KS") or cls_summary.get("ks")
        st.metric("KS", fmt2(ks, 4))
    with col3:
        f1 = cls_summary.get("F1") or cls_summary.get("f1")
        st.metric("F1", fmt2(f1, 4))
    with col4:
        pr_auc = cls_summary.get("PR_AUC") or cls_summary.get("pr_auc")
        st.metric("PR AUC", fmt2(pr_auc, 4))
    
    # Secondary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        precision = cls_summary.get("Precision") or cls_summary.get("precision")
        st.metric("Precision", fmt2(precision, 4))
    with col2:
        recall = cls_summary.get("Recall") or cls_summary.get("recall")
        st.metric("Recall", fmt2(recall, 4))
    with col3:
        accuracy = cls_summary.get("Accuracy") or cls_summary.get("accuracy")
        st.metric("Accuracy", fmt2(accuracy, 4))
    with col4:
        brier = cls_summary.get("Brier") or cls_summary.get("brier")
        st.metric("Brier", fmt2(brier, 4))
    
    # Plots
    cls_plots = results.get("classification_plot_paths", {})
    
    if cls_plots:
        st.divider()
        st.write("**Diagnostic Curves**")
        
        plot_cols = st.columns(2)
        
        plots = [
            ("roc", "ROC Curve"),
            ("pr", "Precision-Recall Curve"),
            ("ks", "KS Curve"),
            ("calibration", "Calibration Plot"),
        ]
        
        for i, (key, label) in enumerate(plots):
            plot_path = cls_plots.get(key)
            if plot_path and os.path.exists(str(plot_path)):
                with plot_cols[i % 2]:
                    st.image(str(plot_path), caption=label, use_container_width=True)


def render_shap_outputs(results: Dict[str, Any]) -> None:
    """
    Render SHAP explainability outputs.
    
    Args:
        results: Results dictionary containing SHAPCheck
    """
    st.subheader("🔍 SHAP Explainability")
    
    shap_check = results.get("SHAPCheck", {})
    if isinstance(shap_check, dict) and "SHAPCheck" in shap_check:
        shap_check = shap_check["SHAPCheck"]
    
    if not shap_check or shap_check.get("skipped"):
        st.info("SHAP analysis was not run or was skipped.")
        return
    
    # Top features
    top_features = shap_check.get("top_features", [])
    if top_features:
        st.write("**Top Contributing Features:**")
        for i, feat in enumerate(top_features[:10], 1):
            st.write(f"{i}. {feat}")
    
    # Plots
    plots = shap_check.get("plots", {})
    
    col1, col2 = st.columns(2)
    
    beeswarm = plots.get("beeswarm") or shap_check.get("shap_plot_path")
    if beeswarm and os.path.exists(beeswarm):
        with col1:
            st.image(beeswarm, caption="SHAP Beeswarm Plot", use_container_width=True)
    
    bar_plot = plots.get("bar")
    if bar_plot and os.path.exists(bar_plot):
        with col2:
            st.image(bar_plot, caption="SHAP Bar Plot", use_container_width=True)


def render_stress_test_outputs(results: Dict[str, Any]) -> None:
    """
    Render stress test results.
    
    Args:
        results: Results dictionary containing StressTestCheck
    """
    st.subheader("💪 Stress Test Results")
    
    stress_check = results.get("StressTestCheck", {})
    if isinstance(stress_check, dict) and "StressTestCheck" in stress_check:
        stress_check = stress_check["StressTestCheck"]
    
    if not stress_check:
        st.info("Stress test was not run.")
        return
    
    table = stress_check.get("table", [])
    if isinstance(stress_check, list):
        table = stress_check
    
    if not table:
        st.info("No stress test results available.")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(table)
    
    if "metric" in df.columns and "baseline" in df.columns and "stressed" in df.columns:
        df["delta"] = df["stressed"] - df["baseline"]
        df["delta_%"] = (df["delta"].abs() / df["baseline"].abs() * 100).round(2)
    
    st.dataframe(df, use_container_width=True)
    
    # Summary
    if "delta" in df.columns:
        max_drop = df["delta"].abs().max()
        if max_drop < 0.02:
            st.success("✅ Model is highly robust - minimal performance change under stress")
        elif max_drop < 0.05:
            st.warning("⚠️ Moderate sensitivity detected under stress conditions")
        else:
            st.error("❌ Significant performance degradation under stress")
