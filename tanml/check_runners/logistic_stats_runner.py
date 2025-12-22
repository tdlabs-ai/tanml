# tanml/check_runners/logistic_stats_runner.py
"""
Logistic stats check runner.

This check fits a statsmodels Logit model as a baseline/challenger
and provides statistical diagnostics including coefficients, p-values,
and confidence intervals.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from tanml.check_runners.base_runner import BaseCheckRunner

# Optional statsmodels import
try:
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
    _HAS_STATSMODELS = True
except ImportError:
    _HAS_STATSMODELS = False


def _is_binary_series(y: pd.Series) -> bool:
    """Check if series has exactly 2 unique values."""
    try:
        u = pd.unique(pd.Series(y).dropna())
        return len(u) == 2
    except Exception:
        return False


def _prep_design_matrix(
    X_like: Any, ref_columns: pd.Index | None, add_const: bool = True
) -> pd.DataFrame:
    """
    Prepare design matrix for statsmodels Logit.
    
    1. Convert to DataFrame
    2. One-hot encode (drop_first=True)
    3. Align to ref_columns if given
    4. Coerce to numeric & sanitize
    5. Optionally add constant
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


class LogisticStatsCheckRunner(BaseCheckRunner):
    """
    Runner for logistic stats challenger model.
    
    Fits a statsmodels Logit on the training data and computes:
    - Coefficient table with standard errors, z-values, p-values, CIs
    - Baseline classification metrics on test data
    - Summary text for reporting
    
    Only runs if the primary model is logistic-like and target is binary.
    
    Output:
        - summary_text: Human-readable statsmodels summary
        - coef_table_rows: Coefficient table as list of dicts
        - baseline_metrics: Test set classification metrics
    """
    
    @property
    def name(self) -> str:
        return "LogisticStatsCheck"
    
    def execute(self) -> Dict[str, Any]:
        """
        Fit Logit model and compute statistics.
        
        Returns:
            Dictionary containing LogitStats results
        """
        if not _HAS_STATSMODELS:
            return {"LogitStats": {"error": "statsmodels not installed"}}
        
        model = self.context.model
        X_train = self.context.X_train
        X_test = self.context.X_test
        y_train = self.context.y_train
        y_test = self.context.y_test
        
        # Skip if model is not logistic-like
        if not self._is_logistic_like(model):
            print("ℹ️ LogisticStatsCheck skipped — model not logistic-like")
            return {"LogitStats": {"skipped": True}}
        
        # Ensure binary target
        y_train_s = pd.Series(y_train)
        if not _is_binary_series(y_train_s):
            print("ℹ️ LogisticStatsCheck skipped — target is not binary")
            return {"LogitStats": {"skipped": True}}
        
        # Encode to 0/1
        counts = y_train_s.value_counts().sort_values(ascending=False).index.tolist()
        enc_map = {counts[0]: 0, counts[1]: 1}
        yb_train = y_train_s.map(enc_map).astype(int)
        
        # Prepare design matrix
        from tanml.checks.logit_stats import _prep_design_matrix_df
        Xd_train = _prep_design_matrix_df(X_train, ref_columns=None, add_const=True)
        
        # Fit statsmodels Logit
        res = sm.Logit(yb_train, Xd_train).fit(disp=0, method="lbfgs", maxiter=1000)
        
        # Summary text
        try:
            summary_text = res.summary2().as_text()
        except Exception:
            summary_text = str(res.summary())
        
        # Coefficient table
        coef_df = self._build_coef_table(res)
        
        # Baseline metrics on test set
        Xd_test = _prep_design_matrix_df(X_test, ref_columns=Xd_train.columns, add_const=True)
        baseline_metrics = self._compute_baseline_metrics(res, Xd_test, y_test, enc_map)
        
        threshold = float(self.get_config_value("threshold", 0.5))
        
        return {
            "LogitStats": {
                "summary_text": summary_text,
                "coef_table_headers": ["feature", "coef", "std err", "z", "P>|z|", "ci_low", "ci_high"],
                "coef_table_rows": coef_df.to_dict(orient="records"),
                "baseline_metrics": baseline_metrics,
                "baseline_note": f"Computed on test split; threshold={threshold}.",
            }
        }
    
    def _is_logistic_like(self, model) -> bool:
        """Check if model is logistic-like."""
        if not _HAS_STATSMODELS:
            return False
        return (
            isinstance(model, LogisticRegression)
            or getattr(model, "__class__", type("X", (object,), {})).__name__.lower().startswith("logit")
            or hasattr(model, "predict_proba")
        )
    
    def _build_coef_table(self, res) -> pd.DataFrame:
        """Build coefficient table from statsmodels result."""
        params = res.params
        bse = res.bse
        zvals = params / bse.replace(0, np.nan)
        pvals = res.pvalues
        ci = res.conf_int(alpha=0.05)
        ci.columns = ["ci_low", "ci_high"]
        
        coef_df = pd.DataFrame({
            "feature": params.index,
            "coef": params.values,
            "std err": bse.values,
            "z": zvals.values,
            "P>|z|": pvals.values,
            "ci_low": ci["ci_low"].values,
            "ci_high": ci["ci_high"].values,
        })
        
        # Put constant first
        if "const" in coef_df["feature"].values:
            coef_df = pd.concat([
                coef_df.loc[coef_df["feature"] == "const"],
                coef_df.loc[coef_df["feature"] != "const"],
            ], ignore_index=True)
        
        # Round numeric columns
        for c in ["coef", "std err", "z", "P>|z|", "ci_low", "ci_high"]:
            coef_df[c] = pd.to_numeric(coef_df[c], errors="coerce").round(4)
        
        return coef_df
    
    def _compute_baseline_metrics(self, res, Xd_test, y_test, enc_map) -> Dict[str, Any]:
        """Compute classification metrics on test set."""
        y_score = res.predict(Xd_test)
        threshold = float(self.get_config_value("threshold", 0.5))
        y_pred = (y_score >= threshold).astype(int)
        
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
        gini = 2 * auc - 1 if (auc == auc) else np.nan
        
        return {
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


# =============================================================================
# Legacy Compatibility
# =============================================================================

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
    """Legacy function interface for LogisticStatsCheck."""
    from tanml.core.context import CheckContext
    
    context = CheckContext(
        model=model,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        config=rule_config,
        cleaned_df=cleaned_df,
    )
    
    runner = LogisticStatsCheckRunner(context)
    return runner.run() or {"LogitStats": {"skipped": True}}
