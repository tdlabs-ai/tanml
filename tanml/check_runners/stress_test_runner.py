from __future__ import annotations
from typing import Any, Dict
import pandas as pd
from tanml.checks.stress_test import StressTestCheck

def run_stress_test_check(model, X_train, X_test, y_train, y_test, rule_config, cleaned_df, *args, **kwargs):
    cfg = (rule_config or {}).get("StressTestCheck", {})
    if not cfg.get("enabled", True):
        print("ℹ️ Skipping StressTestCheck (disabled in rules.yaml)")
        return {"StressTestCheck": {"skipped": True}}

    try:
        epsilon = cfg.get("epsilon", 0.01)            
        perturb_fraction = cfg.get("perturb_fraction", 0.2)

      
        cols_test = getattr(X_test, "columns", None)
        cols_train = getattr(X_train, "columns", None)

        if cols_test is not None:
            columns = list(cols_test)
        elif cols_train is not None:
            columns = list(cols_train)
        else:
            columns = None 

        X_test_df = pd.DataFrame(X_test, columns=columns)

        checker = StressTestCheck(model, X_test_df, y_test, epsilon, perturb_fraction)
        result = checker.run()

        if isinstance(result, list):
            table = result
        elif hasattr(result, "to_dict"):
            table = result.to_dict(orient="records")
        else:
            return {"StressTestCheck": {"output": result}}

        return {"StressTestCheck": {"table": table}}

    except Exception as e:
        print(f"⚠️ StressTestCheck failed: {e}")
        return {"StressTestCheck": {"error": str(e)}}
