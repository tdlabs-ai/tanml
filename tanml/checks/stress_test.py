from __future__ import annotations
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    mean_squared_error, r2_score
)

def _infer_task_type(model, y) -> str:
    # 1) model hint
    if hasattr(model, "_estimator_type"):
        if model._estimator_type == "classifier":
            return "classification"
        if model._estimator_type == "regressor":
            return "regression"
    # 2) label-based fallback
    try:
        s = pd.Series(y).dropna()
        if pd.api.types.is_numeric_dtype(s):
            return "classification" if s.nunique() <= 10 else "regression"
        return "classification"
    except Exception:
        return "classification"

def _scores_for_classification(model, X) -> np.ndarray:
    # Prefer probabilities
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        return p[:, 1] if p.ndim == 2 and p.shape[1] > 1 else np.ravel(p)
    # Fall back to decision scores
    if hasattr(model, "decision_function"):
        return np.ravel(model.decision_function(X))
    # Last resort: hard predictions (will be used directly for acc; AUC may be NaN)
    return np.ravel(model.predict(X))

def _bin_pred_from_score(score: np.ndarray) -> np.ndarray:
    # If looks like probability in [0,1], threshold at 0.5; else at 0.0
    if np.all(np.isfinite(score)):
        smin, smax = float(np.min(score)), float(np.max(score))
        if 0.0 <= smin <= 1.0 and 0.0 <= smax <= 1.0:
            return (score >= 0.5).astype(int)
        return (score >= 0.0).astype(int)
    # fallback
    return (score >= 0.5).astype(int)

def _cls_metrics(y_true, y_score, y_pred) -> Tuple[float, float]:
    acc = float(accuracy_score(y_true, y_pred))
    try:
        auc = float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else np.nan
    except Exception:
        auc = np.nan
    return acc, auc

def _reg_metrics(y_true, y_pred) -> Tuple[float, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    return rmse, r2

class StressTestCheck:
    """
    Task-aware stress test:
      - Classification: accuracy, auc, delta_accuracy, delta_auc
      - Regression:     rmse, r2,  delta_rmse,     delta_r2

    For each numeric feature, perturb a random subset of rows by (1 ± epsilon).
    """

    def __init__(self, model, X, y, epsilon: float = 0.01, perturb_fraction: float = 0.2, random_state: int = 42):
        self.model = model
        self.X = pd.DataFrame(X, columns=getattr(X, "columns", None))
        self.y = np.asarray(y)
        self.epsilon = float(epsilon)
        self.perturb_fraction = float(perturb_fraction)
        self.rng = np.random.default_rng(int(random_state))

        # 🔧 Cast ALL numeric columns to float once to avoid int64→float assignment warnings
        num_cols = [c for c in self.X.columns if is_numeric_dtype(self.X[c]) and not is_bool_dtype(self.X[c])]
        if num_cols:
            self.X[num_cols] = self.X[num_cols].astype("float64")

    def _numeric_cols(self) -> List[str]:
        return [c for c in self.X.columns if is_numeric_dtype(self.X[c]) and not is_bool_dtype(self.X[c])]

    def _perturb_scaled(self, X: pd.DataFrame, col: str, sign: int) -> pd.DataFrame:
        """Scale a random subset of column 'col' by (1 + sign*epsilon)."""
        Xp = X.copy(deep=True)
        if Xp.empty:
            return Xp
        n = len(Xp)
        k = max(1, int(self.perturb_fraction * n))
        idx = self.rng.choice(Xp.index, size=k, replace=False)
        factor = 1.0 + sign * self.epsilon

        # Use a float numpy view for assignment — no dtype warnings
        vals = Xp.loc[idx, col].to_numpy(dtype="float64", copy=False)
        Xp.loc[idx, col] = vals * float(factor)
        return Xp

    def run(self):
        task_type = _infer_task_type(self.model, self.y)
        results: List[Dict[str, Any]] = []

        # ---------- Baseline ----------
        if task_type == "regression":
            y_pred_base = np.ravel(self.model.predict(self.X))
            rmse_base, r2_base = _reg_metrics(self.y, y_pred_base)
        else:
            y_score_base = _scores_for_classification(self.model, self.X)
            # If scores are probs/decision, bin properly; else use model.predict
            try:
                y_pred_base = _bin_pred_from_score(y_score_base)
            except Exception:
                y_pred_base = np.ravel(self.model.predict(self.X))
            acc_base, auc_base = _cls_metrics(self.y, y_score_base, y_pred_base)

        # ---------- Per-feature perturbations ----------
        for col in self._numeric_cols():
            for sign, lab in [(+1, f"+{round(self.epsilon * 100, 2)}%"),
                              (-1, f"-{round(self.epsilon * 100, 2)}%")]:
                try:
                    Xp = self._perturb_scaled(self.X, col, sign)

                    if task_type == "regression":
                        y_pred_p = np.ravel(self.model.predict(Xp))
                        rmse_p, r2_p = _reg_metrics(self.y, y_pred_p)
                        results.append({
                            "feature": col,
                            "perturbation": lab,
                            "rmse": round(rmse_p, 4),
                            "r2": round(r2_p, 4),
                            "delta_rmse": round(rmse_p - rmse_base, 4),
                            "delta_r2": round(r2_p - r2_base, 4),
                        })
                    else:
                        y_score_p = _scores_for_classification(self.model, Xp)
                        try:
                            y_pred_p = _bin_pred_from_score(y_score_p)
                        except Exception:
                            y_pred_p = np.ravel(self.model.predict(Xp))
                        acc_p, auc_p = _cls_metrics(self.y, y_score_p, y_pred_p)
                        results.append({
                            "feature": col,
                            "perturbation": lab,
                            "accuracy": round(acc_p, 4),
                            "auc": round(auc_p, 4) if auc_p == auc_p else np.nan,
                            "delta_accuracy": round(acc_p - acc_base, 4),
                            "delta_auc": round((auc_p - auc_base), 4) if (auc_base == auc_base and auc_p == auc_p) else np.nan,
                        })

                # Robust error row in either mode
                except Exception as e:
                    if task_type == "regression":
                        results.append({
                            "feature": col, "perturbation": lab,
                            "rmse": "error", "r2": "error",
                            "delta_rmse": f"Error: {e}", "delta_r2": f"Error: {e}",
                        })
                    else:
                        results.append({
                            "feature": col, "perturbation": lab,
                            "accuracy": "error", "auc": "error",
                            "delta_accuracy": f"Error: {e}", "delta_auc": f"Error: {e}",
                        })

        return pd.DataFrame(results)
