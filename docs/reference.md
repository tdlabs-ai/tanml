# API Reference

::: tanml.models.registry
    handler: python
    options:
      members:
        - ModelSpec
        - build_estimator
        - list_models
        - get_spec

::: tanml.analysis.drift
    handler: python
    options:
      members:
        - calculate_psi
        - calculate_ks
        - analyze_drift

::: tanml.analysis.correlation
    handler: python
    options:
      members:
        - calculate_correlation_matrix
        - find_highly_correlated_pairs
        - calculate_vif
        - analyze_feature_relationships

::: tanml.analysis.clustering
    handler: python
    options:
      members:
        - analyze_cluster_coverage

::: tanml.checks.stress_test
    handler: python
    options:
      members:
        - StressTestCheck

::: tanml.checks.explainability.shap_check
    handler: python
    options:
      members:
        - SHAPCheck
