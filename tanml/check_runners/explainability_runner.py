# tanml/check_runners/explainability_runner.py

from tanml.checks.explainability.shap_check import SHAPCheck

def run_shap_check(
    model, X_train, X_test, y_train, y_test,
    rule_config, cleaned_df, *args, **kwargs
):
    try:
        cfg = rule_config.get("SHAPCheck", {})
        if not cfg.get("enabled", True):
            print("ℹ️ SHAPCheck skipped (disabled in rules.yaml)")
            return None

        check = SHAPCheck(
            model=model,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            rule_config=rule_config,
            cleaned_df=cleaned_df,
        )
        return check.run()

    except Exception as e:
        print(f"⚠️ SHAPCheck failed: {e}")
        return {"SHAPCheck": {"error": str(e)}}
