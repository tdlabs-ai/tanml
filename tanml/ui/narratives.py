# tanml/ui/narratives.py
"""
Narrative generation helpers for TanML reports.

These functions generate dynamic, human-readable text summaries
of model performance and validation results.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def _get_metric(m: dict[str, Any], key: str) -> float:
    """Case-insensitive dictionary get."""
    for k, v in m.items():
        if k.lower() == key.lower():
            return v
    return 0.0


def story_performance(metrics: dict[str, Any], task_type: str) -> str:
    """
    Generate a dynamic sentence about model performance.

    Args:
        metrics: Dictionary of performance metrics
        task_type: "classification" or "regression"

    Returns:
        Human-readable performance summary
    """
    s = []
    if task_type == "classification":
        auc = _get_metric(metrics, "roc_auc")
        if auc:
            s.append(f"The model achieved an ROC AUC of **{auc:.3f}**.")
            if auc > 0.9:
                s.append("This indicates **excellent** discriminatory power.")
            elif auc > 0.75:
                s.append("This performance is considered **good**.")
            elif auc > 0.6:
                s.append("The model has **moderate** predictive skill.")
            else:
                s.append(
                    "The performance is relatively weak, suggesting more features or data may be needed."
                )

        f1 = _get_metric(metrics, "f1")
        acc = _get_metric(metrics, "accuracy")
        if acc and f1 and abs(acc - f1) > 0.15:
            s.append(
                f"Note the gap between Accuracy ({acc:.2f}) and F1 ({f1:.2f}), suggesting class imbalance issues."
            )

    else:  # Regression
        r2 = _get_metric(metrics, "r2")
        rmse = _get_metric(metrics, "rmse")
        s.append(f"The model captured **{r2:.1%}** of the variance in the target (R2 Score).")
        s.append(f"On average, predictions are off by **{rmse:.3f}** units (RMSE).")
        if r2 > 0.8:
            s.append("This is a **high-precision** model.")
        elif r2 < 0.3:
            s.append(
                "The low R2 suggests the features explain very little of the target's behavior."
            )

    return " ".join(s)


def story_features(metrics_df: pd.DataFrame, top_n: int = 3) -> str:
    """
    Analyze feature dominance from importance metrics.

    Args:
        metrics_df: DataFrame with feature importance scores
        top_n: Number of top features to highlight

    Returns:
        Feature dominance analysis text
    """
    if metrics_df is None or metrics_df.empty:
        return "Feature importance data is not available."

    # Identify the numeric score column
    score_col = None
    for c in ["Power Score", "Composite Score", "importance", "coef", "shap_mean"]:
        if c in metrics_df.columns:
            score_col = c
            break

    if not score_col:
        return "No valid importance score column found."

    sorted_df = metrics_df.sort_values(by=score_col, ascending=False).head(top_n)
    top_names = sorted_df.iloc[:, 0].tolist()  # Assume first col is feature name

    if len(top_names) >= 3:
        return (
            f"The model is heavily influenced by **{top_names[0]}**, **{top_names[1]}**, "
            f"and **{top_names[2]}**. These features contribute the most to predictions."
        )
    elif len(top_names) == 2:
        return f"Key drivers are **{top_names[0]}** and **{top_names[1]}**."
    elif len(top_names) == 1:
        return f"The dominant feature is **{top_names[0]}**."
    return "Could not determine top features."


def story_overfitting(train_m: dict[str, Any], test_m: dict[str, Any]) -> str:
    """
    Check for train-test divergence (overfitting).

    Args:
        train_m: Training metrics dictionary
        test_m: Test metrics dictionary

    Returns:
        Overfitting analysis text
    """
    s = []
    # Normalize keys for matching
    tr_norm = {k.lower(): v for k, v in train_m.items()}
    te_norm = {k.lower(): v for k, v in test_m.items()}

    # Pick a key metric
    keys = ["roc_auc", "r2", "accuracy", "f1", "rmse", "mae"]
    metric = next((k for k in keys if k in tr_norm and k in te_norm), None)

    if metric:
        tr = tr_norm[metric]
        te = te_norm[metric]
        delta = tr - te

        s.append(f"Comparing {metric.upper()}: Train **{tr:.3f}** vs Test **{te:.3f}**.")

        # Logic varies by metric type (Error vs Score)
        is_error = metric in ["rmse", "mae", "log_loss", "brier"]

        # Overfitting check
        problematic = False
        if is_error:
            if delta < -0.1:
                problematic = True
        elif delta > 0.1:
            problematic = True

        if problematic:
            s.append(
                "⚠️ **Significant Overfitting Detected**: The model performs much better on training data than unseen test data. Consider regularization or reducing complexity."
            )
        elif abs(delta) > 0.05:
            s.append(
                "There is **mild divergence** between Train and Test, but it may be within acceptable limits."
            )
        else:
            s.append(
                "The model generalizes well, with **consistent performance** across both datasets."
            )

    else:
        tr_keys = list(train_m.keys()) if train_m else "None"
        s.append(
            f"Could not analyze stability. Metrics not matched in both datasets. Keys found: {str(tr_keys)[:50]}..."
        )

    return " ".join(s)


def story_drift(drift_data) -> str:
    """
    Summarize potential data drift.

    Args:
        drift_data: List of drift analysis results (list of dicts with PSI, Feature, etc.)

    Returns:
        Drift summary text
    """
    if not drift_data:
        return "No drift analysis performed."

    # Handle both list format (from app.py) and dict format
    if isinstance(drift_data, dict):
        # Dict format with aggregated results
        high_drift_features = drift_data.get("high_drift_features", [])
        if not high_drift_features:
            return "✅ **No significant drift detected** between training and test distributions."
        names = ", ".join(high_drift_features[:3])
        extra = f" (+{len(high_drift_features) - 3} more)" if len(high_drift_features) > 3 else ""
        return f"⚠️ **Data drift detected** in {len(high_drift_features)} features: {names}{extra}."

    # List format (original app.py format)
    # Count drifting features
    high_drift = [d for d in drift_data if isinstance(d, dict) and d.get("PSI", 0) > 0.2]
    med_drift = [d for d in drift_data if isinstance(d, dict) and 0.1 < d.get("PSI", 0) <= 0.2]

    if len(high_drift) > 0:
        feats = ", ".join([d.get("Feature", "Unknown") for d in high_drift[:3]])
        return f"🚨 **Critical Drift Alert**: {len(high_drift)} features (including **{feats}**) show significant distribution shifts (PSI > 0.2). Model reliability may be compromised."
    elif len(med_drift) > 0:
        return f"⚠️ **Monitor**: {len(med_drift)} features show slight drift (PSI 0.1-0.2). This is usually acceptable but worth watching."
    else:
        return "✅ **Stable**: No significant data drift detected. The test data distribution closely matches the training data."


def story_stress(stress_data: list[dict[str, Any]]) -> str:
    """
    Analyze robustness under noise from stress test.

    Args:
        stress_data: List of stress test results

    Returns:
        Stress test analysis text
    """
    if not stress_data:
        return "No stress-test data available."

    try:
        # Find largest performance drop
        max_drop = 0
        worst_metric = None

        for row in stress_data:
            baseline = row.get("baseline")
            stressed = row.get("stressed")
            metric = row.get("metric")

            if baseline is not None and stressed is not None:
                drop = abs(baseline - stressed)
                if drop > max_drop:
                    max_drop = drop
                    worst_metric = metric

        if max_drop < 0.02:
            return "✅ **Model is highly robust**: Performance barely changes under stress."
        elif max_drop < 0.05:
            return f"The model shows **moderate resilience**. Largest drop was {max_drop:.3f} in {worst_metric}."
        else:
            return (
                f"⚠️ **Model sensitivity detected**: {worst_metric} dropped by {max_drop:.3f} under noise. "
                "Consider ensemble methods or regularization."
            )
    except Exception:
        return "Could not analyze stress test results."


def story_shap(shap_res: dict[str, Any]) -> str:
    """
    Summarize top drivers from SHAP analysis.

    Args:
        shap_res: SHAP check results

    Returns:
        SHAP summary text
    """
    if not shap_res:
        return "SHAP analysis not available."

    raw_tf = shap_res.get("top_features", [])

    # Clean feature names if they are dicts
    top_features = []
    for item in raw_tf:
        if isinstance(item, dict):
            # Try 'feature' key, else first key
            f_name = item.get("feature")
            if not f_name:
                # Fallback to first key
                f_name = next(iter(item.keys()))
            top_features.append(str(f_name))
        else:
            top_features.append(str(item))

    if not top_features:
        return "SHAP computed but no top features identified."

    if len(top_features) >= 3:
        return (
            f"SHAP analysis identifies **{top_features[0]}**, **{top_features[1]}**, "
            f"and **{top_features[2]}** as the top model drivers."
        )
    elif len(top_features) == 2:
        return f"Top SHAP features: **{top_features[0]}** and **{top_features[1]}**."
    else:
        return f"The dominant SHAP feature is **{top_features[0]}**."


# Legacy aliases for backward compatibility
_story_performance = story_performance
_story_features = story_features
_story_overfitting = story_overfitting
_story_drift = story_drift
_story_stress = story_stress
_story_shap = story_shap
