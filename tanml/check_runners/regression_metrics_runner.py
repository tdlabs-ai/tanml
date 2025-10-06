# tanml/check_runners/regression_metrics_runner.py
from __future__ import annotations

import os
from typing import Any, Dict, Optional

import numpy as np
import matplotlib.pyplot as plt

from tanml.checks.regression_metrics import RegressionMetricsCheck

try:
    from scipy import stats as _scipy_stats
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


# ---------- utils ----------
def _ensure_outdir(config: Dict[str, Any]) -> str:
    base = (config.get("options") or {}).get("save_artifacts_dir") or "reports"
    outdir = os.path.join(base, "regression_metrics")
    os.makedirs(outdir, exist_ok=True)
    return outdir


def _to_1d(x: Any) -> np.ndarray:
    return np.asarray(x).reshape(-1)


def _residuals(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return y_true - y_pred


def _plot_pred_vs_actual(y_true: np.ndarray, y_pred: np.ndarray, save_path: str) -> str:
    plt.figure()
    plt.scatter(y_true, y_pred, s=12, alpha=0.75)
    mn = float(min(np.min(y_true), np.min(y_pred)))
    mx = float(max(np.max(y_true), np.max(y_pred)))
    plt.plot([mn, mx], [mn, mx])  # reference y=x
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Predicted vs Actual")
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()
    return save_path


def _plot_residuals_vs_pred(y_pred: np.ndarray, resid: np.ndarray, save_path: str) -> str:
    plt.figure()
    plt.scatter(y_pred, resid, s=12, alpha=0.75)
    plt.axhline(0.0)
    plt.xlabel("Predicted")
    plt.ylabel("Residual (y_true - y_pred)")
    plt.title("Residuals vs Predicted")
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()
    return save_path


def _plot_residual_hist(resid: np.ndarray, save_path: str) -> str:
    plt.figure()
    plt.hist(resid, bins=30, alpha=0.9)
    plt.xlabel("Residual")
    plt.ylabel("Count")
    plt.title("Residual Distribution")
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()
    return save_path


def _plot_qq(resid: np.ndarray, save_path: str) -> str:
    osm, osr = _scipy_stats.probplot(resid, dist="norm", fit=False)
    plt.figure()
    plt.scatter(osm, osr, s=12, alpha=0.8)
    mn = float(min(np.min(osm), np.min(osr)))
    mx = float(max(np.max(osm), np.max(osr)))
    plt.plot([mn, mx], [mn, mx])
    plt.xlabel("Theoretical Quantiles (Normal)")
    plt.ylabel("Ordered Residuals")
    plt.title("Residuals Q–Q Plot")
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()
    return save_path


def _plot_abs_error_box(abs_err: np.ndarray, save_path: str) -> str:
    plt.figure()
    plt.boxplot(abs_err, vert=True, showfliers=True)
    plt.ylabel("|Residual|")
    plt.title("Absolute Error — Box Plot")
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()
    return save_path


def _plot_abs_error_violin(abs_err: np.ndarray, save_path: str) -> str:
    plt.figure()
    plt.violinplot(abs_err, showmeans=True, showmedians=True)
    plt.ylabel("|Residual|")
    plt.title("Absolute Error — Violin Plot")
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()
    return save_path


def RegressionMetricsCheckRunner(
    model: Any,
    X_train: Any,
    X_test: Any,
    y_train: Any,
    y_test: Any,
    config: Dict[str, Any],
    cleaned_df: Optional[Any] = None,
    raw_df: Optional[Any] = None,
    ctx: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    1) Predict on X_test
    2) Compute frozen regression metrics
    3) Save 5 standard charts (Q–Q skipped if SciPy missing)
    4) Return structured results for engine/report
    """
    # 1) predictions
    try:
        y_pred = model.predict(X_test)
    except Exception as e:
        raise RuntimeError(f"Model prediction failed in RegressionMetricsCheckRunner: {e}")

    y_true = _to_1d(y_test)
    y_pred = _to_1d(y_pred)

    # n_features for Adjusted R²
    try:
        n_features = int(getattr(X_train, "shape", [None, None])[1])
    except Exception:
        n_features = None

    # 2) metrics
    chk = RegressionMetricsCheck(
        y_true=y_true,
        y_pred=y_pred,
        n_features=n_features,
        config=(config or {}),
    )
    metrics = chk.run()

    # 3) plots
    outdir = _ensure_outdir(config)
    resid = _residuals(y_true, y_pred)
    abs_err = np.abs(resid)

    p1 = os.path.join(outdir, "pred_vs_actual.png")
    p2 = os.path.join(outdir, "residuals_vs_pred.png")
    p3 = os.path.join(outdir, "residual_hist.png")
    p4 = os.path.join(outdir, "qq_plot.png")
    b1 = os.path.join(outdir, "abs_error_box.png")
    v1 = os.path.join(outdir, "abs_error_violin.png")

    try:
        _plot_pred_vs_actual(y_true, y_pred, p1)
        _plot_residuals_vs_pred(y_pred, resid, p2)
        _plot_residual_hist(resid, p3)

        if _HAS_SCIPY:
            _plot_qq(resid, p4)
        else:
            (metrics.get("notes") or []).append("Q–Q plot skipped: SciPy not available.")
            p4 = None

        _plot_abs_error_box(abs_err, b1)
        _plot_abs_error_violin(abs_err, v1)
    except Exception as e:
        (metrics.get("notes") or []).append(f"Plotting failed: {e}")

    # 4) return
    return {
        "RegressionMetrics": {
            **metrics,
            "artifacts": {
                "pred_vs_actual": p1,
                "residuals_vs_pred": p2,
                "residual_hist": p3,
                "qq_plot": p4,            
                "abs_error_box": b1,
                "abs_error_violin": v1,
            },
        }
    }
