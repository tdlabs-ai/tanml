# tanml/check_runners/cleaning_repro_runner.py
from tanml.checks.cleaning_repro import CleaningReproCheck

def run_cleaning_repro_check(model, X_train, X_test, y_train, y_test,
                             config, cleaned_data, *args, **kwargs):
    # honour rules.yaml toggle
    if not config.get("rules", {}).get("CleaningReproCheck", {}).get("enabled", True):
        print("ℹ️ CleaningReproCheck skipped (disabled in rules.yaml)")
        return None

    # raw_df can come from rules.yaml *or* via kwargs (passed by ValidationEngine)
    raw_data = config.get("raw_data") or kwargs.get("raw_df")
    if raw_data is None:
        print("⚠️ Skipping CleaningReproCheck — raw_data missing in config and kwargs")
        return {"CleaningReproCheck": {"error": "raw_data not available"}}

    check = CleaningReproCheck(raw_data, cleaned_data)
    return {"CleaningReproCheck": check.run()}
