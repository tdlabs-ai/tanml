# explainability runner
from tanml.checks.explainability.shap_check import SHAPCheck

def run_shap_check(model, X_train, X_test, y_train, y_test, rule_config, cleaned_df, *args, **kwargs):
    try:
        cfg_shapcheck = (rule_config or {}).get("SHAPCheck", {}) or {}
        cfg_expl = (rule_config or {}).get("explainability", {}).get("shap", {}) or {}
        enabled = cfg_shapcheck.get("enabled", cfg_expl.get("enabled", True))
        if not enabled:
            print("ℹ️ SHAPCheck skipped (disabled)")
            return {"SHAPCheck": {"skipped": True}}

        check = SHAPCheck(model, X_train, X_test, y_train, y_test, rule_config=rule_config, cleaned_df=cleaned_df)
        result = check.run()
        return {"SHAPCheck": result}
    except Exception as e:
        print(f"⚠️ SHAPCheck failed: {e}")
        return {"SHAPCheck": {"status": "error", "error": str(e)}}
