# tanml/ui/services/cv.py
"""
Cross-validation helper for TanML UI.

This module contains the _run_repeated_cv function extracted from app.py.
"""

from __future__ import annotations


def _run_repeated_cv(model, X, y, task_type, n_splits=5, n_repeats=5, seed=42):
    """
    Runs RepeatedStratifiedKFold (clf) or RepeatedKFold (reg) and returns
    a dict of stats suitable for report injection (mean, std, p05, p50, p95).
    """
    import numpy as np
    from sklearn.base import clone
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        brier_score_loss,
        f1_score,
        log_loss,
        matthews_corrcoef,
        mean_absolute_error,
        mean_squared_error,
        median_absolute_error,
        precision_score,
        r2_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold
    from sklearn.preprocessing import LabelEncoder

    # Decide CV strategy
    if task_type == "classification":
        cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
        # Ensure y is proper for stratification
        # If string labels, encode them temporarily for CV split generation
        try:
            le = LabelEncoder()
            y_enc = le.fit_transform(y)
        except:
            y_enc = y
    else:
        cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
        y_enc = y

    # Metrics storage
    scores = {}

    def _add(k, v):
        scores.setdefault(k, []).append(v)

    # Pre-check if model handles probabilities
    has_proba = hasattr(model, "predict_proba")

    X_arr = X.values if hasattr(X, "values") else np.array(X)
    y_arr = y.values if hasattr(y, "values") else np.array(y)

    # Ensure arrays are numeric (handle edge cases with SAS/SPSS data)
    try:
        X_arr = X_arr.astype(np.float64)
    except (ValueError, TypeError):
        # If conversion fails, try to select only numeric columns
        if hasattr(X, "select_dtypes"):
            X_numeric = X.select_dtypes(include=[np.number])
            X_arr = X_numeric.values.astype(np.float64)
        else:
            X_arr = np.array(X, dtype=np.float64)

    # Handle y conversion
    try:
        y_arr = y_arr.astype(np.float64)
    except (ValueError, TypeError):
        # For classification with string labels, keep as is
        pass

    for train_idx, test_idx in cv.split(X_arr, y_enc):
        X_tr, X_val = X_arr[train_idx], X_arr[test_idx]
        y_tr, y_val = y_arr[train_idx], y_arr[test_idx]

        m = clone(model)
        m.fit(X_tr, y_tr)

        y_pred = m.predict(X_val)

        if task_type == "classification":
            y_prob = m.predict_proba(X_val)[:, 1] if has_proba else None

            if y_prob is not None:
                try:
                    _add("roc_auc", roc_auc_score(y_val, y_prob))
                except:
                    pass
                try:
                    _add("pr_auc", average_precision_score(y_val, y_prob))
                except:
                    pass
                try:
                    _add("brier", brier_score_loss(y_val, y_prob))
                except:
                    pass
                try:
                    _add("log_loss", log_loss(y_val, y_prob))
                except:
                    pass
                try:
                    _add("gini", 2 * roc_auc_score(y_val, y_prob) - 1)
                except:
                    pass
                try:
                    p0 = y_prob[y_val == 0]
                    p1 = y_prob[y_val == 1]
                    from scipy.stats import ks_2samp

                    _add("ks", ks_2samp(p0, p1).statistic)
                except:
                    pass

                # --- Curve Data Collection for Spaghetti Plots ---
                from sklearn.metrics import precision_recall_curve, roc_curve

                if "curves" not in scores:
                    scores["curves"] = {"roc": [], "pr": []}
                try:
                    fpr, tpr, th_roc = roc_curve(y_val, y_prob)
                    scores["curves"]["roc"].append((fpr, tpr, th_roc))
                except:
                    pass
                try:
                    prec, rec, th_pr = precision_recall_curve(y_val, y_prob)
                    scores["curves"]["pr"].append(
                        (rec, prec, th_pr)
                    )  # Note: recall on x, precision on y typically
                except:
                    pass

                # --- Pooled OOF Data for Lift/Gain/Confusion ---
                if "oof" not in scores:
                    scores["oof"] = {"y_true": [], "y_prob": [], "y_pred": []}
                scores["oof"]["y_true"].extend(y_val.tolist())
                scores["oof"]["y_prob"].extend(y_prob.tolist())
                scores["oof"]["y_pred"].extend(y_pred.tolist())

                # --- Per-Fold Arrays for CDF KS Plot ---
                if "y_probs" not in scores:
                    scores["y_probs"] = []
                if "y_trues" not in scores:
                    scores["y_trues"] = []
                scores["y_probs"].append(y_prob)
                scores["y_trues"].append(y_val)

            try:
                _add("f1", f1_score(y_val, y_pred))
            except:
                pass
            try:
                _add("precision", precision_score(y_val, y_pred))
            except:
                pass
            try:
                _add("recall", recall_score(y_val, y_pred))
            except:
                pass
            try:
                _add("accuracy", accuracy_score(y_val, y_pred))
            except:
                pass
            try:
                _add("bal_acc", balanced_accuracy_score(y_val, y_pred))
            except:
                pass
            try:
                _add("mcc", matthews_corrcoef(y_val, y_pred))
            except:
                pass

        else:
            try:
                _add("rmse", np.sqrt(mean_squared_error(y_val, y_pred)))
            except:
                pass
            try:
                _add("mae", mean_absolute_error(y_val, y_pred))
            except:
                pass
            try:
                _add("median_ae", median_absolute_error(y_val, y_pred))
            except:
                pass
            try:
                _add("r2", r2_score(y_val, y_pred))
            except:
                pass

            # --- Pooled OOF Data for Predicted vs Actual ---
            if "oof" not in scores:
                scores["oof"] = {"y_true": [], "y_pred": []}
            scores["oof"]["y_true"].extend(y_val.tolist())
            scores["oof"]["y_pred"].extend(y_pred.tolist())

    # Compute Aggregates
    stats_out = {}
    # Skip keys that contain variable-length arrays
    skip_keys = ["curves", "oof", "y_probs", "y_trues"]
    for k, v_list in scores.items():
        if k in skip_keys:
            continue
        arr = np.array(v_list)
        stats_out[k] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "p05": float(np.percentile(arr, 5)),
            "p50": float(np.median(arr)),
            "p95": float(np.percentile(arr, 95)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "raw": v_list,  # Keep raw scores for plotting
        }

    if "curves" in scores:
        stats_out["curves"] = scores["curves"]

    if "oof" in scores:
        stats_out["oof"] = scores["oof"]

    if "y_probs" in scores:
        stats_out["y_probs"] = scores["y_probs"]

    if "y_trues" in scores:
        stats_out["y_trues"] = scores["y_trues"]

    stats_out["threshold_info"] = {"rule": "Default (0.5)", "value": 0.5}
    return stats_out
