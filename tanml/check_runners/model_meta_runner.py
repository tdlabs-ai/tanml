from tanml.checks.model_meta import ModelMetaCheck

def ModelMetaCheckRunner(model, X_train, X_test, y_train, y_test, rule_config, cleaned_df, *args, **kwargs):
    try:
        cfg = rule_config.get("ModelMetaCheck", {})
        if not cfg.get("enabled", True):
            print("ℹ️ ModelMetaCheck skipped (disabled in rules.yaml)")
            return None

        check = ModelMetaCheck(
            model=model,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            rule_config=rule_config,
            cleaned_data=cleaned_df
        )
        return check.run()

    except Exception as e:
        print(f"⚠️ ModelMetaCheck failed: {e}")
        return {"ModelMetaCheck": {"error": str(e)}}
