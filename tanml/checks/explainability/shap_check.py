# tanml/checks/explainability/shap_check.py
from tanml.checks.base import BaseCheck

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import shap
import numpy as np
import pandas as pd
import traceback
import warnings
from pathlib import Path
from datetime import datetime
from scipy import sparse as sp

def _safe_numeric_cast_df(df: pd.DataFrame) -> pd.DataFrame:
    """Convert columns to numeric where possible; leave others unchanged."""
    out = {}
    for c in df.columns:
        s = df[c]
        try:
            out[c] = pd.to_numeric(s)  # no 'errors' kwarg
        except Exception:
            out[c] = s
    return pd.DataFrame(out, index=df.index)


class SHAPCheck(BaseCheck):
    """
    SHAP for regression + binary classification (no multiclass).

    Config under rule_config["explainability"]["shap"]:
      - enabled: bool (default True, also checked under rule_config["SHAPCheck"]["enabled"])
      - task: "auto" | "classification" | "regression"          (default "auto")
      - algorithm: "auto" | "tree" | "linear" | "kernel" | "permutation" (default "auto")
      - model_output: "auto" | "raw" | "log_odds" | "probability"  (tree-only hint; default "auto")
      - background_strategy: "sample" | "kmeans"                 (default "sample")
      - background_sample_size: int                              (default 100)
      - test_sample_size: int                                    (default 200)
      - max_display: int                                         (default 20)
      - seed: int                                                (default 42)
      - out_dir: str (optional)                                  (preferred save folder)
    """

    def __init__(self, model, X_train, X_test, y_train, y_test, rule_config=None, cleaned_df=None):
        super().__init__(model, X_train, X_test, y_train, y_test, rule_config, cleaned_data=cleaned_df)
        self.cleaned_df = cleaned_df

    # -------------------------- helpers --------------------------

    @staticmethod
    def _to_df(X, names=None):
        if isinstance(X, pd.DataFrame):
            return X
        if sp.issparse(X):
            df = pd.DataFrame.sparse.from_spmatrix(X)
        else:
            df = pd.DataFrame(np.asarray(X))
        if names is not None and len(names) == df.shape[1]:
            df.columns = list(names)
        return df

    @staticmethod
    def _task(y, forced="auto"):
        if forced and forced != "auto":
            return forced
        try:
            yv = y.iloc[:, 0] if isinstance(y, pd.DataFrame) else (y if isinstance(y, pd.Series) else pd.Series(y))
            uniq = pd.Series(yv).dropna().unique()
            return "classification" if len(uniq) <= 2 else "regression"
        except Exception:
            return "regression"

    @staticmethod
    def _pos_cls_idx(model, X_one):
        """Return index of the positive class (1/True if available, else max-label) for binary classification."""
        try:
            if hasattr(model, "classes_") and len(model.classes_) == 2:
                classes = list(model.classes_)
                for pos in (1, True):
                    if pos in classes:
                        return classes.index(pos)
                return classes.index(max(classes))
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_one)
                return 1 if proba.shape[1] == 2 else 0
        except Exception:
            pass
        return 1

    @staticmethod
    def _looks_like_tree(m):
        mod = type(m).__module__.lower()
        name = type(m).__name__.lower()
        return (
            "xgboost" in mod
            or "lightgbm" in mod
            or "catboost" in mod
            or "sklearn.ensemble" in mod
            or "sklearn.tree" in mod
            or "randomforest" in name
            or "gradientboost" in name
            or "extratrees" in name
            or "decisiontree" in name
        )

    @staticmethod
    def _looks_like_linear(m):
        mod = type(m).__module__.lower()
        name = type(m).__name__.lower()
        return (
            "sklearn.linear_model" in mod
            or "logistic" in name
            or "linear" in name
            or "ridge" in name
            or "lasso" in name
            or "elastic" in name
        )

    def _predict_fn(self, is_cls: bool, pos_idx: int | None):
        """
        Vectorized prediction function for permutation/kernel explainers.
        Returns positive-class probability when classification is detected and predict_proba is available.
        """
        if is_cls and hasattr(self.model, "predict_proba"):
            def f(X):
                p = self.model.predict_proba(X)
                i = 1 if (p.ndim == 2 and p.shape[1] == 2) else (pos_idx or 0)
                return p[:, i]
            return f
        return self.model.predict

    def _explainer(self, algorithm, background, model_output_hint, is_cls, pos_idx):
        """
        Choose fastest viable explainer:
          - tree → TreeExplainer (with interventional perturbation)
          - linear → LinearExplainer
          - default/auto → PermutationExplainer (avoid slow Kernel, unless explicitly requested)
        """
        m = self.model
        alg = (algorithm or "auto").lower()

        # Prefer fast paths in auto
        if alg == "tree" or (alg == "auto" and self._looks_like_tree(m)):
            mo = None if model_output_hint == "auto" else model_output_hint
            expl = shap.TreeExplainer(
                m, data=background, feature_perturbation="interventional", model_output=mo
            )
            return expl, "tree"

        if alg == "linear" or (alg == "auto" and self._looks_like_linear(m)):
            return shap.LinearExplainer(m, background), "linear"

        if alg == "permutation" or alg == "auto":
            fn = self._predict_fn(is_cls, pos_idx)
            return shap.explainers.Permutation(fn, background, max_evals=2000), "perm"

        # Only use Kernel if explicitly requested
        if alg == "kernel":
            fn = self._predict_fn(is_cls, pos_idx)
            return shap.KernelExplainer(fn, background), "kernel"

        # Fallback (should not hit)
        fn = self._predict_fn(is_cls, pos_idx)
        return shap.explainers.Permutation(fn, background, max_evals=2000), "perm"

    # ---------------------------- main ----------------------------

    def run(self):
        out = {}
        try:
            warnings.filterwarnings("ignore", category=UserWarning)

            # -------- read config (note: under explainability.shap) ----------
            exp_cfg = (self.rule_config or {}).get("explainability", {}) or {}
            cfg = exp_cfg.get("shap", {}) if isinstance(exp_cfg, dict) else {}
            seed = int(cfg.get("seed", 42))
            bg_n = int(cfg.get("background_sample_size", 100))
            test_n = int(cfg.get("test_sample_size", 200))
            task_forced = (cfg.get("task") or "auto").lower()
            algorithm = (cfg.get("algorithm") or "auto").lower()
            bg_strategy = (cfg.get("background_strategy") or "sample").lower()
            model_output_hint = (cfg.get("model_output") or "auto").lower()
            max_display = int(cfg.get("max_display", 20))

            # -------- resolve output directory + timestamp ----------
            out_dir_opt = cfg.get("out_dir")
            options_dir = ((self.rule_config or {}).get("options") or {}).get("save_artifacts_dir")
            # prefer explicit shap.out_dir, then global artifacts dir, then local fallback
            outdir = Path(out_dir_opt or options_dir or (Path(__file__).resolve().parents[2] / "tmp_report_assets"))
            outdir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")

            # -------- materialize dataframes & coerce dtypes ----------
            feature_names = list(self.X_train.columns) if isinstance(self.X_train, pd.DataFrame) else None
            Xtr = self._to_df(self.X_train, feature_names)
            Xte = self._to_df(self.X_test, feature_names)
            if Xtr.empty or Xte.empty:
                raise ValueError("Empty X_train or X_test for SHAP.")

            # Avoid slow implicit object->float conversions during plotting
            Xtr = _safe_numeric_cast_df(Xtr)
            Xte = _safe_numeric_cast_df(Xte)


            # -------- task resolution & sanity for binary classification ----------
            task = self._task(self.y_train, forced=task_forced)
            is_cls = (task == "classification")
            if is_cls:
                yv = self.y_train if isinstance(self.y_train, (pd.Series, pd.DataFrame)) else pd.Series(self.y_train)
                if len(pd.Series(yv).dropna().unique()) > 2:
                    raise ValueError("Binary classification only: y_train has >2 classes.")

            # positive-class index hint (used for permutation/kernel predict function)
            pos_idx_hint = self._pos_cls_idx(self.model, Xte.iloc[:1].values)

            # -------- background selection ----------
            if bg_strategy == "kmeans" and len(Xtr) > bg_n and not sp.issparse(self.X_train):
                background = shap.kmeans(Xtr, bg_n, seed=seed)
            else:
                background = shap.utils.sample(Xtr, bg_n, random_state=seed)

            # -------- slice test rows to explain ----------
            Xs = Xte.head(test_n)

            # -------- choose explainer & compute SHAP once ----------
            explainer, kind = self._explainer(algorithm, background, model_output_hint, is_cls, pos_idx_hint)
            if kind == "tree":
                sv = explainer(Xs, check_additivity=False)  # big speedup, visually identical plots
            else:
                sv = explainer(Xs)

            bg_shape = background.shape if hasattr(background, "shape") else None
            print(f"SHAP explainer={type(explainer).__name__} kind={kind} Xs={Xs.shape} "
                  f"bg={'kmeans' if bg_shape is None else bg_shape}")

            # -------- squeeze to 2-D for binary cls (if needed) ----------
            vals = sv.values
            if hasattr(vals, "ndim") and vals.ndim == 3:
                pos_idx = self._pos_cls_idx(self.model, Xs.iloc[:1].values)
                sv.values = vals[:, :, pos_idx]
                if isinstance(sv.base_values, np.ndarray) and sv.base_values.ndim == 2:
                    sv.base_values = sv.base_values[:, pos_idx]
            else:
                pos_idx = None if task == "regression" else self._pos_cls_idx(self.model, Xs.iloc[:1].values)

            # -------- save plots ----------
            segment = "global"

            beeswarm_path = outdir / f"shap_beeswarm_{segment}_{ts}.png"
            plt.figure(figsize=(9, 6))
            shap.plots.beeswarm(sv, max_display=max_display, show=False)
            plt.tight_layout()
            plt.savefig(beeswarm_path, bbox_inches="tight", dpi=120, transparent=False)
            plt.close()

            bar_path = outdir / f"shap_bar_{segment}_{ts}.png"
            plt.figure(figsize=(9, 6))
            shap.plots.bar(sv, max_display=max_display, show=False)
            plt.tight_layout()
            plt.savefig(bar_path, bbox_inches="tight", dpi=120, transparent=False)
            plt.close()

            # -------- top features ----------
            # mean absolute SHAP across rows
            mean_abs = np.abs(sv.values).mean(axis=0)
            idx = np.argsort(mean_abs)[::-1][:max_display]
            cols = list(Xs.columns)
            top = [{"feature": cols[i] if i < len(cols) else f"f{i}", "mean_abs_shap": float(mean_abs[i])} for i in idx]
            top_list_pairs = [[d["feature"], d["mean_abs_shap"]] for d in top]  # compat with old report builder

            out.update({
                "status": "ok",
                "task": task,
                "positive_class_index": pos_idx if task == "classification" else None,
                # new + old keys (backward-compatible)
                "plots": {"beeswarm": str(beeswarm_path), "bar": str(bar_path)},
                "images": {"beeswarm": str(beeswarm_path), "bar": str(bar_path)},
                "top_features": top,
                "shap_top_features": top_list_pairs,
            })
            print(f"✅ SHAP saved: {beeswarm_path}, {bar_path}")

        except Exception:
            out["status"] = "error: " + traceback.format_exc()
        return out
