from tanml.checks.stress_test import StressTestCheck

def run_stress_test_check(model, X_train, X_test, y_train, y_test, rule_config, cleaned_df, *args, **kwargs):
    cfg = rule_config.get("StressTestCheck", {})
    if not cfg.get("enabled", True):
        print("ℹ️ Skipping StressTestCheck (disabled in rules.yaml)")
        return {"StressTestCheck": {"skipped": True}}

    try:
        epsilon = cfg.get("epsilon", 0.01)
        perturb_fraction = cfg.get("perturb_fraction", 0.2)

        checker = StressTestCheck(model, X_test, y_test, epsilon, perturb_fraction)
        result = checker.run()

        # Ensure output is always a dictionary
        if isinstance(result, list):
            return {"StressTestCheck": {"table": result}}
        elif hasattr(result, "to_dict"):
            return {"StressTestCheck": {"table": result.to_dict(orient="records")}}
        else:
            return {"StressTestCheck": {"output": result}}

    except Exception as e:
        print(f"⚠️ StressTestCheck failed: {e}")
        return {"StressTestCheck": {"error": str(e)}}
