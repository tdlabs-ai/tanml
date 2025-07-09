from tanml.checks.data_quality import DataQualityCheck

def run_data_quality_check(model, X_train, X_test, y_train, y_test, rule_config, cleaned_df, *args, **kwargs):
    dq_cfg = rule_config.get("DataQualityCheck", {})
    if not dq_cfg.get("enabled", True):
        print("ℹ️ Skipping DataQualityCheck (disabled in rules.yaml)")
        return {"DataQualityCheck": {"skipped": True}}

    try:
        check = DataQualityCheck(
            model=model,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            rule_config=dq_cfg,
            cleaned_data=cleaned_df
        )
        result = check.run()
        return {"DataQualityCheck": result}

    except Exception as e:
        print(f"⚠️ DataQualityCheck failed: {e}")
        return {"DataQualityCheck": {"error": str(e)}}
