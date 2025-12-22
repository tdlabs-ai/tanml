# tanml/ui/services/validation.py
"""
Validation orchestration service for TanML UI.

Provides high-level API for running validation checks and
generating reports.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from tanml.engine.core_engine_agent import ValidationEngine
from tanml.report.report_builder import ReportBuilder


def run_validation(
    model: Any,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    config: Dict[str, Any],
    cleaned_df: pd.DataFrame,
    raw_df: Optional[pd.DataFrame] = None,
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> Dict[str, Any]:
    """
    Run full model validation.
    
    Args:
        model: Trained model
        X_train: Training features
        X_test: Test features
        y_train: Training target
        y_test: Test target
        config: Validation configuration
        cleaned_df: Cleaned DataFrame
        raw_df: Optional raw DataFrame for comparison
        progress_callback: Optional callback(message, progress) for updates
        
    Returns:
        Dictionary containing all validation results
    """
    if progress_callback:
        progress_callback("Initializing validation engine...", 0.1)
    
    engine = ValidationEngine(
        model=model,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        config=config,
        cleaned_data=cleaned_df,
        raw_data=raw_df,
    )
    
    if progress_callback:
        progress_callback("Running validation checks...", 0.3)
    
    results = engine.run_all_checks()
    
    if progress_callback:
        progress_callback("Validation complete!", 1.0)
    
    return results


def generate_report(
    results: Dict[str, Any],
    template_path: Path,
    output_path: Path,
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> Path:
    """
    Generate a validation report document.
    
    Args:
        results: Validation results dictionary
        template_path: Path to report template
        output_path: Path for output document
        progress_callback: Optional callback for progress updates
        
    Returns:
        Path to the generated report
    """
    if progress_callback:
        progress_callback("Generating report...", 0.5)
    
    builder = ReportBuilder(
        results=results,
        template_path=template_path,
        output_path=output_path,
    )
    
    builder.build()
    
    if progress_callback:
        progress_callback("Report generated!", 1.0)
    
    return Path(output_path)


def run_validation_with_progress(
    model: Any,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    config: Dict[str, Any],
    cleaned_df: pd.DataFrame,
    raw_df: Optional[pd.DataFrame] = None,
) -> Tuple[Dict[str, Any], st.container]:
    """
    Run validation with Streamlit progress display.
    
    Args:
        model: Trained model
        X_train, X_test, y_train, y_test: Train/test splits
        config: Validation configuration
        cleaned_df: Cleaned DataFrame
        raw_df: Optional raw DataFrame
        
    Returns:
        Tuple of (results, progress_container)
    """
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    def update_progress(message: str, progress: float):
        progress_bar.progress(progress)
        status_text.text(message)
    
    results = run_validation(
        model=model,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        config=config,
        cleaned_df=cleaned_df,
        raw_df=raw_df,
        progress_callback=update_progress,
    )
    
    progress_bar.empty()
    status_text.empty()
    
    return results


def get_check_summary(results: Dict[str, Any]) -> Dict[str, str]:
    """
    Get a summary of check statuses.
    
    Args:
        results: Validation results
        
    Returns:
        Dictionary of check_name -> status ("ok", "warning", "error", "skipped")
    """
    summary = {}
    
    check_keys = [
        "EDACheck", "CorrelationCheck", "VIFCheck", "PerformanceCheck",
        "ModelMetaCheck", "StressTestCheck", "SHAPCheck", "DataQualityCheck",
        "RuleEngineCheck", "RawDataCheck", "InputClusterCheck",
    ]
    
    for key in check_keys:
        check_result = results.get(key, {})
        
        if not check_result:
            summary[key] = "skipped"
        elif isinstance(check_result, dict):
            if check_result.get("error"):
                summary[key] = "error"
            elif check_result.get("skipped"):
                summary[key] = "skipped"
            else:
                summary[key] = "ok"
        else:
            summary[key] = "ok"
    
    return summary
