from tanml.check_runners.performance_runner import run_performance_check
from tanml.check_runners.data_quality_runner import run_data_quality_check
from tanml.check_runners.stress_test_runner import run_stress_test_check
from tanml.check_runners.input_cluster_runner import run_input_cluster_check
from tanml.check_runners.logistic_stats_runner import run_logistic_stats_check
from tanml.check_runners.raw_data_runner import run_raw_data_check
#from tanml.check_runners.cleaning_repro_runner import run_cleaning_repro_check
from tanml.check_runners.model_meta_runner import ModelMetaCheckRunner
from tanml.check_runners.correlation_runner import CorrelationCheckRunner
from tanml.check_runners.eda_runner import EDACheckRunner
from tanml.check_runners.explainability_runner import run_shap_check
from tanml.check_runners.vif_runner import VIFCheckRunner
from tanml.check_runners.rule_engine_runner import RuleEngineCheckRunner


# Wrapper for InputClusterCheck to inject expected_features from model
def input_cluster_wrapper(model, X_train, X_test, y_train, y_test, rule_config, cleaned_df, *args, **kwargs):
    if hasattr(model, "feature_names_in_"):
        expected_features = list(model.feature_names_in_)
    else:
        raise ValueError("Model does not have 'feature_names_in_' attribute required for InputClusterCheck.")
    
    return run_input_cluster_check(
        model, X_train, X_test, y_train, y_test, rule_config, cleaned_df, expected_features
    )

CHECK_RUNNER_REGISTRY = {
    "PerformanceCheck": run_performance_check,
    "DataQualityCheck": run_data_quality_check,
    "StressTestCheck": run_stress_test_check,
    "InputClusterCheck": input_cluster_wrapper,  
    "LogisticStatsCheck": run_logistic_stats_check,
    "RawDataCheck": run_raw_data_check,
    #"CleaningReproCheck": run_cleaning_repro_check,
    "ModelMetaCheck": ModelMetaCheckRunner,  
    "CorrelationCheck": CorrelationCheckRunner, 
    "EDACheck": EDACheckRunner,  
    "SHAPCheck": run_shap_check,
    "VIFCheck": VIFCheckRunner,
    "RuleEngineCheck": RuleEngineCheckRunner, 

}
