# tanml/engine/core_engine_agent.py

from tanml.engine.check_agent_registry import CHECK_RUNNER_REGISTRY

import numpy as np
import pandas as pd

try:
    import statsmodels.api as sm
    _SM_AVAILABLE = True
except Exception:
    _SM_AVAILABLE = False

KEEP_AS_NESTED = {
    "DataQualityCheck",
    "StressTestCheck",
    "InputClusterCheck",
    "InputClusterCoverageCheck",
    "RawDataCheck",
    "SHAPCheck",
    "VIFCheck",
    "CorrelationCheck",
    "EDACheck",
    # "RegressionMetrics",  # you can keep nested if desired
}


class ValidationEngine:
    def __init__(
        self,
        model,
        X_train,
        X_test,
        y_train,
        y_test,
        config,
        cleaned_data,
        raw_df=None,
        ctx=None
    ):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.config = config
        self.cleaned_data = cleaned_data
        self.raw_df = raw_df

        # allow resuming if config had check_results
        self.results = dict(config.get("check_results", {}))
        self.ctx = ctx or {}

        self.task_type = self._infer_task_type(self.y_train, config, model)

    # --- better detection logic --------------------------------------------
    @staticmethod
    def _infer_task_type(y, config=None, model=None):
        """
        Decide if task is classification or regression.
        Priority:
        1. config["model"]["type"]
        2. model._estimator_type (sklearn)
        3. y values (unique count)
        """
        # 1. Config hint
        try:
            mtype = (config or {}).get("model", {}).get("type", "")
            if isinstance(mtype, str):
                mtype = mtype.lower()
                if "class" in mtype:
                    return "classification"
                if "regress" in mtype:
                    return "regression"
        except Exception:
            pass

        # 2. Model introspection
        try:
            if hasattr(model, "_estimator_type"):
                est = getattr(model, "_estimator_type", "")
                if est == "classifier":
                    return "classification"
                if est == "regressor":
                    return "regression"
            if hasattr(model, "predict_proba") or hasattr(model, "decision_function"):
                return "classification"
        except Exception:
            pass

        # 3. Label based
        try:
            if isinstance(y, (pd.Series, pd.DataFrame)):
                s = y.squeeze()
            else:
                s = np.asarray(y).reshape(-1)

            unique_vals = pd.Series(s).dropna().unique()
            # Heuristic: small discrete set -> classification
            if pd.api.types.is_numeric_dtype(s):
                if len(unique_vals) <= 10:
                    return "classification"
                return "regression"
            else:
                # non-numeric target -> classification
                return "classification"
        except Exception:
            pass

        # Fallback
        return "classification"
    # -----------------------------------------------------------------------

    def _pick(self, *paths, default=None):
        for path in paths:
            cur = self.results
            ok = True
            for p in path:
                if isinstance(cur, dict) and p in cur:
                    cur = cur[p]
                else:
                    ok = False
                    break
            if ok:
                return cur
        return default

    def _compute_linear_stats(self):
        """
        Optional: compute a statsmodels OLS summary + coefficient table for regression runs.
        Writes results into self.results["LinearStats"].
        """
        if self.task_type != "regression":
            return
        if not _SM_AVAILABLE:
            self.results["LinearStats"] = {
                "error": "statsmodels not available; install `statsmodels` to see OLS summary."
            }
            return

        try:
            # add constant and fit OLS on TRAIN split to mirror sklearn fit
            X = self.X_train
            y = self.y_train
            Xc = sm.add_constant(X, has_constant="add")
            ols_model = sm.OLS(y, Xc, missing="drop")
            ols_res = ols_model.fit()

            # Build coefficient table (including intercept 'const')
            params = ols_res.params
            bse = ols_res.bse
            tvals = ols_res.tvalues
            pvals = ols_res.pvalues
            ci = ols_res.conf_int(alpha=0.05)
            ci.columns = ["ci_low", "ci_high"]

            rows = []
            for name in params.index:
                rows.append({
                    "feature": name,
                    "coef": float(params[name]),
                    "std err": float(bse.get(name, float("nan"))),
                    "t": float(tvals.get(name, float("nan"))),
                    "P>|t|": float(pvals.get(name, float("nan"))),
                    "ci_low": float(ci.loc[name, "ci_low"]) if name in ci.index else None,
                    "ci_high": float(ci.loc[name, "ci_high"]) if name in ci.index else None,
                })

            self.results["LinearStats"] = {
                "summary_text": ols_res.summary().as_text(),
                "coeff_table": rows,
                "status": "ok",
            }
        except Exception as e:
            self.results["LinearStats"] = {"error": f"OLS stats failed: {e}"}
    # ------------------------------------------------------------

    def run_all_checks(self, progress_callback=None):
        self.results["task_type"] = self.task_type

        for check_name, runner_func in CHECK_RUNNER_REGISTRY.items():
            if check_name in self.config.get("skip_checks", []):
                continue

            if progress_callback:
                try:
                    progress_callback(f"Running {check_name}…")
                except Exception:
                    pass

            print(f"✅ Running {check_name}")
            try:
                result = runner_func(
                    self.model,
                    self.X_train,
                    self.X_test,
                    self.y_train,
                    self.y_test,
                    self.config,
                    self.cleaned_data,
                    raw_df=self.raw_df
                )
                self._integrate(check_name, result)
            except Exception as e:
                print(f"⚠️  {check_name} failed: {e}")
                self.results[check_name] = {"error": str(e)}

        # add OLS stats for regression (pretty coef table + p-values)
        self._compute_linear_stats()

        # -------- Build summary: TASK-AWARE --------
        summary = {}

        if self.task_type == "regression":
            summary["rmse"] = self._pick(("RegressionMetrics", "rmse"))
            summary["mae"]  = self._pick(("RegressionMetrics", "mae"))
            summary["r2"]   = self._pick(("RegressionMetrics", "r2"))
        else:
            cls = self._pick(("performance", "classification", "summary")) or {}
            summary["auc"]    = cls.get("auc")
            summary["ks"]     = cls.get("ks")
            summary["f1"]     = cls.get("f1")
            summary["pr_auc"] = cls.get("pr_auc")

        # PSI (optional)
        summary["max_psi"] = self._pick(
            ("PSICheck", "max_psi"),
            ("PopulationStabilityCheck", "max_psi"),
            ("max_psi",)
        )

        # Count failed checks
        failed = 0
        for k, v in self.results.items():
            if isinstance(v, dict) and v.get("status") == "fail":
                failed += 1
        summary["rules_failed"] = failed

        self.results["summary"] = summary
        self.results["check_results"] = dict(self.results)
        return self.results

    def _integrate(self, check_name: str, result):
        """Merge a check result into self.results respecting the template layout."""
        if not result:
            return

        if check_name == "LogisticStatsCheck":
            self.results.update(result)
            return

        if not isinstance(result, dict):
            self.results[check_name] = result
            return

        cluster_aliases = {
            "InputClusterCoverageCheck",
            "InputClusterCoverage",
            "ClusterCoverageCheck",
            "InputClustersCheck",
        }
        if check_name in cluster_aliases:
            self.results[check_name] = result
            self.results["InputClusterCheck"] = result
            return

        if set(result.keys()) == {"InputClusterCheck"}:
            self.results["InputClusterCheck"] = result["InputClusterCheck"]
            return

        if check_name in KEEP_AS_NESTED:
            self.results[check_name] = result
            return

        if isinstance(result, dict):
            self.results.update(result)
            return
