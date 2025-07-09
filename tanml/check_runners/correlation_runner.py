from tanml.checks.correlation import CorrelationCheck

def CorrelationCheckRunner(model, X_train, X_test, y_train, y_test, rule_config, cleaned_df, *args, **kwargs):
    try:
        cfg = rule_config.get("CorrelationCheck", {})
        if not cfg.get("enabled", True):
            print("ℹ️ CorrelationCheck skipped (disabled in rules.yaml)")
            return None

        check = CorrelationCheck(cleaned_df)
        return check.run()

    except Exception as e:
        print(f"⚠️ CorrelationCheck failed: {e}")
        return {"CorrelationCheck": {"error": str(e)}}
