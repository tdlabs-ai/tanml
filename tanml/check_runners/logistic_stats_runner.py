from tanml.checks.logit_stats import LogisticStatsCheck
from sklearn.linear_model import LogisticRegression

def run_logistic_stats_check(model, X_train, X_test, y_train, y_test, rule_config, cleaned_df, *args, **kwargs):
    try:
        # Check if model is a LogisticRegression or statsmodels object
        is_logistic = isinstance(model, LogisticRegression) or (
            hasattr(model, "llf") and hasattr(model, "params") and hasattr(model, "summary")
        )

        if not is_logistic:
            print("ℹ️ LogisticStatsCheck skipped — model is not logistic or not recognized")
            return None

        # Use training data only for fitting stats
        check = LogisticStatsCheck(model, X_train, y_train, rule_config)
        output = check.run()

        return {
            "LogisticStatsCheck": output["table"],
            "LogisticStatsFit": output["fit"],
            "LogisticStatsSummary": output["summary"],
            "LogisticStatsCheck_obj": output["object"]
        }

    except Exception as e:
        print(f"⚠️ LogisticStatsCheck failed: {e}")
        return {"LogisticStatsCheck": {"error": str(e)}}
