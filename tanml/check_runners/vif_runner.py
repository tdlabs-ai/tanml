# tanml/check_runners/vif_runner.py
from __future__ import annotations

import os
from pathlib import Path
import pandas as pd

from tanml.utils.data_loader import load_dataframe
from tanml.checks.vif import VIFCheck  


def VIFCheckRunner(
    model, X_train, X_test, y_train, y_test,
    rule_config, cleaned_df, *args, **kwargs
):
    """
    Ensure cleaned_df is a DataFrame; if a path (csv/xlsx/parquet/etc.),
    load it via the universal loader, then run VIFCheck.
    """
    # 1) Normalize cleaned_df to a DataFrame
    try:
        if isinstance(cleaned_df, (str, bytes, os.PathLike, Path)):
            cleaned_df = load_dataframe(cleaned_df)
        elif not isinstance(cleaned_df, pd.DataFrame):
            msg = "cleaned_df is not a DataFrame or loadable path; skipping VIF."
            print(f"ℹ️ {msg}")
            return {"vif_table": [], "high_vif_features": [], "error": msg}
    except Exception as e:
        err = f"Could not load cleaned_df: {e}"
        print(f"⚠️ {err}")
        return {"vif_table": [], "high_vif_features": [], "error": err}

    # 2) Run the check
    try:
        check = VIFCheck(model, X_train, X_test, y_train, y_test, rule_config, cleaned_df)
        result = check.run()  
        # 3) Normalize result
        if isinstance(result, dict) and "vif_table" in result:
            vif_rows = result["vif_table"]
        elif isinstance(result, list):
            vif_rows = result
        else:
            raise ValueError("Unexpected VIFCheck return shape")

        # 4) Canonicalize keys and values
        for row in vif_rows:
            if "Feature" not in row and "feature" in row:
                row["Feature"] = row.pop("feature")
            if "VIF" in row and row["VIF"] is not None:
                row["VIF"] = round(float(row["VIF"]), 2)

        # 5) Identify high VIF features
        threshold = rule_config.get("vif_threshold", 5)
        high_vif = [r["Feature"] for r in vif_rows if r.get("VIF") is not None and r["VIF"] > threshold]

        return {"vif_table": vif_rows, "high_vif_features": high_vif, "error": None}

    except Exception as e:
        print(f"⚠️ VIFCheck failed: {e}")
        return {"vif_table": [], "high_vif_features": [], "error": str(e)}
