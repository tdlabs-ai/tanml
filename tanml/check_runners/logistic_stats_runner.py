# tanml/check_runners/logistic_stats_runner.py
from __future__ import annotations

from typing import Any, Dict
from tanml.checks.logit_stats import _prep_design_matrix_df  
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_fscore_support,
    accuracy_score,
    average_precision_score,
    brier_score_loss,
)


def _is_binary_series(y: pd.Series) -> bool:
    try:
        u = pd.unique(pd.Series(y).dropna())
        return len(u) == 2
    except Exception:
        return False


def _prep_design_matrix(
    X_like: Any, ref_columns: pd.Index | None, add_const: bool = True
) -> pd.DataFrame:
    """
    1) Convert to DataFrame
    2) One-hot encode (drop_first=True)
    3) Align to ref_columns (if given), filling missing cols with 0 and dropping extras
    4) Coerce to numeric & sanitize
    5) Optionally add constant
    """
    Xd = X_like if isinstance(X_like, pd.DataFrame) else pd.DataFrame(X_like)
    Xd = pd.get_dummies(Xd, drop_first=True)

    if ref_columns is not None:
        ref_wo_const = [c for c in ref_columns if c != "const"]
        Xd = Xd.reindex(columns=ref_wo_const, fill_value=0.0)

    for c in Xd.columns:
        Xd[c] = pd.to_numeric(Xd[c], errors="coerce")
    Xd = Xd.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if add_const:
        Xd = sm.add_constant(Xd, has_constant="add")

    return Xd


def run_logistic_stats_check(
    model,
    X_train,
    X_test,
    y_train,
    y_test,
    rule_config: Dict[str, Any],
    cleaned_df,
    *args,
    **kwargs,
) -> Dict[str, Any]:
    """
    Logistic challenger (stats-only):

    - Fits a statsmodels Logit on a one-hot design of X_train (with intercept)
    - Produces: summary_text and coefficient table with CIs
    - Computes baseline classification metrics on the test set (NO plots/CSVs)

    Returns:
    {
      "LogitStats": {
        "summary_text": str,
        "coef_table_headers": ["feature","coef","std err","z","P>|z|","ci_low","ci_high"],
        "coef_table_rows": [ {...}, ... ],
        "baseline_metrics": { "summary": {...} },  # rounded, no 'plots'/'tables'
        "baseline_note": "..."
      }
    }
    """
    try:
        # 1) Skip if model is obviously not logistic-like
        is_logistic_like = (
            isinstance(model, LogisticRegression)
            or getattr(model, "__class__", type("X", (object,), {})).__name__.lower().startswith("logit")
            or hasattr(model, "predict_proba")
        )
        if not is_logistic_like:
            print("ℹ️ LogisticStatsCheck skipped — model not logistic-like")
            return {"LogitStats": {"skipped": True}}

        # 2) Ensure binary target
        y_train_s = pd.Series(y_train)
        if not _is_binary_series(y_train_s):
            print("ℹ️ LogisticStatsCheck skipped — target is not binary")
            return {"LogitStats": {"skipped": True}}

        # Robust 0/1 encoding (majority -> 0, minority -> 1)
        counts = y_train_s.value_counts().sort_values(ascending=False).index.tolist()
        enc_map = {counts[0]: 0, counts[1]: 1}
        yb_train = y_train_s.map(enc_map).astype(int)

        # 3) Train design matrix (with intercept)
        Xd_train = _prep_design_matrix_df(X_train, ref_columns=None, add_const=True)  

        # 4) Fit statsmodels Logit (MLE)
        res = sm.Logit(yb_train, Xd_train).fit(disp=0, method="lbfgs", maxiter=1000)

        # 5) Summary text (human-readable)
        try:
            summary_text = res.summary2().as_text()
        except Exception:
            summary_text = str(res.summary())

        # 6) Coefficient table (const first)
        params = res.params
        bse = res.bse
        # Avoid divide-by-zero in z; replace zeros with NaN then fill after rounding
        zvals = params / bse.replace(0, np.nan)
        pvals = res.pvalues
        ci = res.conf_int(alpha=0.05)
        ci.columns = ["ci_low", "ci_high"]

        coef_df = pd.DataFrame(
            {
                "feature": params.index,
                "coef": params.values,
                "std err": bse.values,
                "z": zvals.values,
                "P>|z|": pvals.values,
                "ci_low": ci["ci_low"].values,
                "ci_high": ci["ci_high"].values,
            }
        )

        if "const" in coef_df["feature"].values:
            coef_df = pd.concat(
                [
                    coef_df.loc[coef_df["feature"] == "const"],
                    coef_df.loc[coef_df["feature"] != "const"],
                ],
                ignore_index=True,
            )

        for c in ["coef", "std err", "z", "P>|z|", "ci_low", "ci_high"]:
            coef_df[c] = pd.to_numeric(coef_df[c], errors="coerce").round(4)

        # 7) Test-set baseline metrics (NO PLOTS/CSVs)
        #    Build test matrix aligned to the training design columns.
        Xd_test = _prep_design_matrix_df(X_test, ref_columns=Xd_train.columns, add_const=True)  

        # Statsmodels Logit returns probability for class "1"
        y_score = res.predict(Xd_test)  # shape (n_test,)

        # Threshold policy (aligned with PerformanceCheck if present)
        threshold = (rule_config.get("PerformanceCheck", {}) or {}).get("threshold", 0.5)
        try:
            thr = float(threshold)
        except Exception:
            thr = 0.5

        y_pred = (y_score >= thr).astype(int)

        yb_test = pd.Series(y_test).map(enc_map).astype(int).to_numpy()

        has_posneg = len(np.unique(yb_test)) > 1
        auc = roc_auc_score(yb_test, y_score) if has_posneg else np.nan
        fpr, tpr, _ = roc_curve(yb_test, y_score) if has_posneg else (np.array([]), np.array([]), None)
        ks = float(np.max(np.abs(tpr - fpr))) if len(fpr) else np.nan
        ap = average_precision_score(yb_test, y_score) if has_posneg else np.nan
        brier = brier_score_loss(yb_test, y_score)
        precision, recall, f1, _ = precision_recall_fscore_support(
            yb_test, y_pred, average="binary", pos_label=1, zero_division=0
        )
        acc = accuracy_score(yb_test, y_pred)
        gini = 2 * auc - 1 if (auc == auc) else np.nan  # handle NaN

        baseline_metrics = {
            "summary": {
                "auc": None if auc != auc else round(float(auc), 2),
                "ks": None if ks != ks else round(float(ks), 2),
                "accuracy": round(float(acc), 2),
                "precision": round(float(precision), 2),
                "recall": round(float(recall), 2),
                "f1": round(float(f1), 2),
                "pr_auc": None if ap != ap else round(float(ap), 2),
                "brier": round(float(brier), 2),
                "gini": None if gini != gini else round(float(gini), 2),
            }
        }

        return {
            "LogitStats": {
                "summary_text": summary_text,
                "coef_table_headers": ["feature", "coef", "std err", "z", "P>|z|", "ci_low", "ci_high"],
                "coef_table_rows": coef_df.to_dict(orient="records"),
                "baseline_metrics": baseline_metrics,  # <-- metrics only; no plots/tables
                "baseline_note": f"Computed on the same test split and preprocessing as the primary model; threshold={thr}.",
            }
        }

    except Exception as e:
        print(f"⚠️ LogisticStatsCheck failed: {e}")
        return {"LogitStats": {"error": str(e)}}
