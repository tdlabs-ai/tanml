from __future__ import annotations
import os
from typing import Any, Dict
import numpy as np
from tanml.checks.performance_classification import compute_classification_report

def _resolve_outdir(config: Dict[str, Any]) -> str:
    base = (config.get("paths") or {}).get("artifacts_dir") \
        or (config.get("options") or {}).get("save_artifacts_dir") \
        or "reports"
    outdir = os.path.join(base, "performance")
    os.makedirs(outdir, exist_ok=True)
    return outdir

def PerformanceCheckRunner(
    model,
    X_train, X_test, y_train, y_test,
    config: Dict[str, Any],
    cleaned_df,
    raw_df=None,
    ctx=None,
):
    outdir = _resolve_outdir(config)
    task_type = ((config.get("model") or {}).get("type") or "binary_classification").lower()

    payload: Dict[str, Any] = {}

    if "class" in task_type: 
        # --- build scores ---
        def _scores(m, X):
            if hasattr(m, "predict_proba"):
                p = m.predict_proba(X)
                return p[:, 1] if p.ndim == 2 and p.shape[1] > 1 else p.ravel()
            if hasattr(m, "decision_function"):
                return m.decision_function(X).ravel()
            return m.predict(X).ravel()

        y_score = _scores(model, X_test)
        y_pred  = getattr(model, "predict")(X_test)

        cls_dir = os.path.join(outdir, "classification")
        os.makedirs(cls_dir, exist_ok=True)

        results_cls = compute_classification_report(
            y_true=np.asarray(y_test),
            y_score=np.asarray(y_score),
            y_pred=np.asarray(y_pred),
            outdir=cls_dir,
            pos_label=1,
            title_prefix=(config.get("model") or {}).get("name", "Model"),
        )

        payload = {
            "performance": {
                "classification": results_cls
            },
            "task_type": "classification",
        }

    else:
       
        # payload = {"performance": {"regression": results_reg}, "task_type": "regression"}
        payload = {"task_type": "regression"}

    return payload

# ---- Back-compat alias so registry can import old name ----
def run_performance_check(
    model,
    X_train, X_test, y_train, y_test,
    config,
    cleaned_df,
    raw_df=None,
    ctx=None,
):
    return PerformanceCheckRunner(
        model=model,
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
        config=config,
        cleaned_df=cleaned_df,
        raw_df=raw_df,
        ctx=ctx,
    )
