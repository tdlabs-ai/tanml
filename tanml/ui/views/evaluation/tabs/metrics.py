# tanml/ui/pages/evaluation/tabs/metrics.py
"""
Metrics Comparison Tab - Compares train vs test metrics.
"""

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from tanml.ui.views.evaluation.tabs import register_tab


@register_tab(name="Metrics Comparison", order=10, key="tab_metrics")
def render(context):
    """Render the metrics comparison tab."""

    # Definitions helper
    with st.expander("ℹ️ Metric Definitions (Click to expand)"):
        st.markdown("""
        | Metric | Definition |
        | :--- | :--- |
        | **Accuracy** | Fraction of correct predictions. (Higher is better) |
        | **Precision** | Out of all predicted positives, how many were actually positive? (Focus: minimizing false positives) |
        | **Recall** | Out of all actual positives, how many did we catch? (Focus: minimizing false negatives) |
        | **F1** | Harmonic mean of Precision and Recall. Balances both concerns. |
        | **AUC (ROC)** | Area Under Curve. Probability that the model ranks a random positive example higher than a negative one. (0.5 = random, 1.0 = perfect) |
        | **RMSE** | Root Mean Squared Error. Standard deviation of prediction errors. (Lower is better) |
        | **MAE** | Mean Absolute Error. Average absolute difference between prediction and actual. (Lower is better) |
        | **R²** | Coefficient of Determination. 1.0 means perfect fit, 0.0 means model is as good as guessing the mean. |
        """)

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
