# tanml/check_runners/vif_runner.py

from tanml.checks.vif import VIFCheck
import pandas as pd
from pathlib import Path

def VIFCheckRunner(
    model, X_train, X_test, y_train, y_test,
    rule_config, cleaned_df, *args, **kwargs
):
    # Ensure cleaned_df is a DataFrame
    if isinstance(cleaned_df, (str, Path)):
        try:
            cleaned_df = pd.read_csv(cleaned_df)
        except Exception as e:
            err = f"Could not read cleaned_df CSV: {e}"
            print(f"⚠️ {err}")
            return {"vif_table": [], "high_vif_features": [], "error": err}

    try:
        check = VIFCheck(model, X_train, X_test, y_train, y_test, rule_config, cleaned_df)
        result = check.run()  # Could be dict or list

        # Normalize result regardless of format
        if isinstance(result, dict) and "vif_table" in result:
            vif_rows = result["vif_table"]
        elif isinstance(result, list):
            vif_rows = result
        else:
            raise ValueError("Unexpected VIFCheck return shape")

        # Rename 'feature' to 'Feature', round VIF values
        for row in vif_rows:
            if "Feature" not in row and "feature" in row:
                row["Feature"] = row.pop("feature")
            row["VIF"] = round(float(row["VIF"]), 2)

        # Identify high VIF features
        threshold = rule_config.get("vif_threshold", 5)
        high_vif = [
            row["Feature"] for row in vif_rows
            if row.get("VIF") is not None and row["VIF"] > threshold
        ]

        # Return final output
        return {
            "vif_table": vif_rows,
            "high_vif_features": high_vif,
            "error": None,
        }

    except Exception as e:
        print(f"⚠️ VIFCheck failed: {e}")
        return {"vif_table": [], "high_vif_features": [], "error": str(e)}
