from tanml.checks.raw_data import RawDataCheck
import pandas as pd

def run_raw_data_check(model, X_train, X_test, y_train, y_test,
                       rule_config, cleaned_data, *args, **kwargs):
    try:
        # ---- locate raw data (DF or path) -------------
        raw_obj = (
            rule_config.get("raw_data") or
            rule_config.get("paths", {}).get("raw_data")
        )
        if raw_obj is None:
            print("ℹ️ RawDataCheck skipped — raw_data not provided in config.")
            return None

        # CSV path → load once
        if isinstance(raw_obj, (str, bytes)):
            raw_obj = pd.read_csv(raw_obj)

        if not isinstance(raw_obj, pd.DataFrame):
            print("ℹ️ RawDataCheck skipped — raw_data is not a DataFrame.")
            return None

        # ---- run the check -----------------------------
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
        return stats["RawDataCheck"]   # hand the inner dict to ValidationEngine

    except Exception as e:
        print(f"⚠️ RawDataCheck failed: {e}")
        return {"error": str(e)}
