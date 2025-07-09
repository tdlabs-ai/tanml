from tanml.checks.performance import PerformanceCheck

def run_performance_check(model, X_train, X_test, y_train, y_test, rule_config, cleaned_df, *args, **kwargs):
    perf_cfg = rule_config.get("PerformanceCheck", {})
    if not perf_cfg.get("enabled", True):
        print("ℹ️ Skipping PerformanceCheck (disabled in rules.yaml)")
        return {"PerformanceCheck": {"skipped": True}}

    if X_test is None or y_test is None or len(X_test) == 0 or len(y_test) == 0:
        print("⚠️ Skipping PerformanceCheck due to empty test data.")
        return {"PerformanceCheck": {"error": "Test data is empty :skipping performance evaluation."}}

    try:
        check = PerformanceCheck(
            model=model,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            rule_config=perf_cfg,
            cleaned_data=cleaned_df
        )
        result = check.run()
        return {"PerformanceCheck": result}

    except Exception as e:
        print(f"⚠️ PerformanceCheck failed: {e}")
        return {"PerformanceCheck": {"error": str(e)}}
