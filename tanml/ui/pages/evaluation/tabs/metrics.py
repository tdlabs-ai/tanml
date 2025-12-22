# tanml/ui/pages/evaluation/tabs/metrics.py
"""
Metrics Comparison Tab - Compares train vs test metrics.
"""

import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, mean_squared_error, mean_absolute_error, r2_score
)
import numpy as np

from tanml.ui.pages.evaluation.tabs import register_tab


@register_tab(name="Metrics Comparison", order=10, key="tab_metrics")
def render(context):
    """Render the metrics comparison tab."""
    
    # Calculate metrics based on task type
    if context.task_type == "classification":
        metrics_order = ["Accuracy", "Precision", "Recall", "F1", "AUC"]
        
        scores_tr = {
            "Accuracy": accuracy_score(context.y_train, context.y_pred_train),
            "Precision": precision_score(context.y_train, context.y_pred_train, zero_division=0),
            "Recall": recall_score(context.y_train, context.y_pred_train, zero_division=0),
            "F1": f1_score(context.y_train, context.y_pred_train, zero_division=0),
        }
        scores_te = {
            "Accuracy": accuracy_score(context.y_test, context.y_pred_test),
            "Precision": precision_score(context.y_test, context.y_pred_test, zero_division=0),
            "Recall": recall_score(context.y_test, context.y_pred_test, zero_division=0),
            "F1": f1_score(context.y_test, context.y_pred_test, zero_division=0),
        }
        
        # Add AUC if probabilities available
        if context.y_prob_train is not None and context.y_prob_test is not None:
            try:
                scores_tr["AUC"] = roc_auc_score(context.y_train, context.y_prob_train)
                scores_te["AUC"] = roc_auc_score(context.y_test, context.y_prob_test)
            except:
                pass
    else:
        # Regression
        metrics_order = ["RMSE", "MAE", "R2"]
        scores_tr = {
            "RMSE": np.sqrt(mean_squared_error(context.y_train, context.y_pred_train)),
            "MAE": mean_absolute_error(context.y_train, context.y_pred_train),
            "R2": r2_score(context.y_train, context.y_pred_train),
        }
        scores_te = {
            "RMSE": np.sqrt(mean_squared_error(context.y_test, context.y_pred_test)),
            "MAE": mean_absolute_error(context.y_test, context.y_pred_test),
            "R2": r2_score(context.y_test, context.y_pred_test),
        }
    
    # Store results for report
    context.results["metrics_train"] = scores_tr
    context.results["metrics_test"] = scores_te
    
    # Display
    c_m1, c_m2 = st.columns(2)
    
    def _render_metrics(col, title, s_dict):
        col.write(f"**{title}**")
        res = []
        for m_name in metrics_order:
            if m_name in s_dict:
                res.append({"Metric": m_name, "Score": s_dict[m_name]})
        # Fallback for any extra metrics
        for m_name, score in s_dict.items():
            if m_name not in metrics_order:
                res.append({"Metric": m_name, "Score": score})
        
        if res:
            col.dataframe(pd.DataFrame(res).style.format({"Score": "{:.4f}"}))
        else:
            col.write("No metrics.")
    
    _render_metrics(c_m1, "TRAIN Metrics", scores_tr)
    _render_metrics(c_m2, "TEST Metrics", scores_te)
