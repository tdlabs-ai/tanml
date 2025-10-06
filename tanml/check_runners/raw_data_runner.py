from __future__ import annotations

import os
import pandas as pd

from tanml.checks.raw_data import RawDataCheck
from tanml.utils.data_loader import load_dataframe


def run_raw_data_check(
    model,
    X_train,
    X_test,
    y_train,
    y_test,
    rule_config,
    cleaned_data,
    *args,
    **kwargs
):
    try:
        # Locate raw data from config
        raw_obj = (
            rule_config.get("raw_data")
            or (rule_config.get("paths", {}) or {}).get("raw_data")
            or (rule_config.get("paths", {}) or {}).get("raw")
        )

        if raw_obj is None:
            print("ℹ️ RawDataCheck skipped — raw_data not provided in config.")
            return None

        if isinstance(raw_obj, (str, bytes, os.PathLike)):
            raw_obj = load_dataframe(raw_obj)

        if not isinstance(raw_obj, pd.DataFrame):
            print("ℹ️ RawDataCheck skipped — raw_data is not a DataFrame or loadable path.")
            return None

        # Run the check
        check = RawDataCheck(
            model=model,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            rule_config=rule_config,
            cleaned_data=cleaned_data,
            raw_data=raw_obj
        )
        stats = check.run()
        return stats.get("RawDataCheck", stats)

    except Exception as e:
        print(f"⚠️ RawDataCheck failed: {e}")
        return {"error": str(e)}
