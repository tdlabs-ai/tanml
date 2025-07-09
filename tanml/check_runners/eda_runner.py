from tanml.checks.eda import EDACheck

def EDACheckRunner(
    model, X_train, X_test, y_train, y_test,
    rule_config, cleaned_df, *args, **kwargs
):
    try:
        cfg = rule_config.get("EDACheck", {})
        if not cfg.get("enabled", True):
            print("ℹ️ EDACheck skipped (disabled in rules.yaml)")
            return None

        check = EDACheck(
            cleaned_data=cleaned_df,
            rule_config=rule_config  
        )
        return check.run()

    except Exception as e:
        print(f"⚠️ EDACheck failed: {e}")
        return {"EDACheck": {"error": str(e)}}
