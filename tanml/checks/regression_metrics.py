# tanml/checks/regression_metrics.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

try:
    # Prefer sklearn implementations when available
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False

from .base import BaseCheck


class RegressionMetricsCheck(BaseCheck):
    """
    Computes TanML's frozen regression metrics:
      - RMSE
      - MAE
      - Median Absolute Error
      - R²
      - Adjusted R²
      - MAPE (or SMAPE fallback when zeros/near-zeros exist in y_true)

    Pure compute: no file I/O, no plotting. Returns a dict.
    """

    def __init__(
        self,
        *,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        n_features: Optional[int] = None,
        mape_eps: float = 1e-8,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
        y_pred : array-like of shape (n_samples,)
        n_features : int, optional
            Number of model features (for Adjusted R²). If None, Adjusted R² may be None.
        mape_eps : float
            Small constant to guard divisions in MAPE/SMAPE.
        config : dict, optional
            Reserved for future options.
        """
        self.y_true = np.asarray(y_true).reshape(-1)
        self.y_pred = np.asarray(y_pred).reshape(-1)
        self.n_features = int(n_features) if n_features is not None else None
        self.mape_eps = float(mape_eps)
        self.config = config or {}

        self._notes: List[str] = []

    # ---------------------------
    # Public API
    # ---------------------------
    def run(self) -> Dict[str, Any]:
        self._validate_inputs()

        rmse = self._rmse(self.y_true, self.y_pred)
        mae = self._mae(self.y_true, self.y_pred)
        median_ae = self._median_ae(self.y_true, self.y_pred)

        r2, r2_adj = self._r2_and_adjusted(self.y_true, self.y_pred, self.n_features)

        mape_val, smape_val, used = self._mape_or_smape(self.y_true, self.y_pred, self.mape_eps)

        return {
            "rmse": rmse,
            "mae": mae,
            "median_ae": median_ae,
            "r2": r2,
            "r2_adjusted": r2_adj,
            "mape_or_smape": mape_val if used == "MAPE" else smape_val,
            "mape_used": (used == "MAPE"),
            "notes": self._notes,  # human-readable notes (e.g., SMAPE fallback, zero variance)
        }

    # ---------------------------
    # Internals
    # ---------------------------
    def _validate_inputs(self) -> None:
        if self.y_true.shape != self.y_pred.shape:
            raise ValueError(f"Shapes differ: y_true{self.y_true.shape} vs y_pred{self.y_pred.shape}")
        if self.y_true.ndim != 1 or self.y_pred.ndim != 1:
            raise ValueError("y_true and y_pred must be 1-D arrays.")
        if self.y_true.size < 2:
            self._notes.append("Too few samples (<2) — some metrics may be undefined.")

    @staticmethod
    def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if _HAS_SKLEARN:
            return float(np.sqrt(mean_squared_error(y_true, y_pred)))
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    @staticmethod
    def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if _HAS_SKLEARN:
            return float(mean_absolute_error(y_true, y_pred))
        return float(np.mean(np.abs(y_true - y_pred)))

    @staticmethod
    def _median_ae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.median(np.abs(y_true - y_pred)))

    def _r2_and_adjusted(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        n_features: Optional[int],
    ) -> Tuple[Optional[float], Optional[float]]:
        n = int(y_true.size)
        # If variance is zero, R² is undefined
        if float(np.var(y_true)) == 0.0:
            self._notes.append("R² undefined: target has zero variance.")
            return None, None

        if _HAS_SKLEARN:
            try:
                r2_val = float(r2_score(y_true, y_pred))
            except Exception:
                r2_val = None
                self._notes.append("R² could not be computed via sklearn.r2_score.")
        else:
            ss_res = float(np.sum((y_true - y_pred) ** 2))
            ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
            r2_val = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else None

        if r2_val is None or n_features is None:
            return r2_val, None

        p = int(n_features)
        if n <= p + 1:
            self._notes.append("Adjusted R² unavailable: insufficient degrees of freedom (n <= p + 1).")
            return r2_val, None

        try:
            r2_adj = float(1.0 - (1.0 - r2_val) * (n - 1) / (n - p - 1))
        except Exception:
            r2_adj = None
            self._notes.append("Adjusted R² computation failed due to numeric issues.")
        return r2_val, r2_adj

    def _mape_or_smape(
        self, y_true: np.ndarray, y_pred: np.ndarray, eps: float
    ) -> Tuple[Optional[float], Optional[float], str]:
        """Return (MAPE, SMAPE, used_flag) and record notes for fallbacks."""
        has_near_zero = np.any(np.abs(y_true) <= eps)
        smape_val = self._smape(y_true, y_pred, eps)

        if has_near_zero:
            self._notes.append("MAPE skipped due to zeros/near-zeros in target; SMAPE reported instead.")
            return None, smape_val, "SMAPE"

        mape_val = float(np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100.0)
        return mape_val, smape_val, "MAPE"

    @staticmethod
    def _smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float) -> float:
        num = 2.0 * np.abs(y_pred - y_true)
        den = np.abs(y_true) + np.abs(y_pred) + eps
        return float(np.mean(num / den) * 100.0)
