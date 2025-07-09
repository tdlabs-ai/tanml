
# -----------------------------------------------------------------------------
# File: tanml/check_runners/input_cluster_runner.py
# -----------------------------------------------------------------------------
from pathlib import Path
from typing import Any, Dict

from tanml.checks.input_cluster import InputClusterCoverageCheck


def run_input_cluster_check(
    model,
    X_train,
    X_test,
    y_train,
    y_test,
    rule_config,
    cleaned_df,
    expected_features,
    *args,
    **kwargs,
):

    cfg = rule_config.get("InputClusterCoverageCheck", {})
    if not cfg.get("enabled", False):
        print("ℹ️  InputClusterCoverageCheck skipped (disabled in rules.yaml)")
        return {"skipped": True}

    try:
        result = InputClusterCoverageCheck(
            cleaned_df=cleaned_df,
            feature_names=expected_features,
            rule_config=rule_config,
        ).run()
        return result

    except Exception as e:
        print(f"⚠️  InputClusterCoverageCheck failed: {e}")
        return {"error": str(e)}

    except Exception as e:  
        print(f"⚠️  InputClusterCoverageCheck failed: {e}")
        return {"InputClusterCheck": {"error": str(e)}}
