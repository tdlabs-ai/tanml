# tanml/ui/services/rule_config.py
"""
Rule configuration builder for TanML validation.

Builds the configuration dictionary used by the validation engine.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional


def build_rule_config(
    *,
    # Data paths
    saved_raw: Optional[Path] = None,
    artifacts_dir: Path,
    
    # Thresholds
    auc_min: float = 0.0,
    f1_min: float = 0.0,
    ks_min: float = 0.0,
    
    # Check toggles
    eda_enabled: bool = True,
    eda_max_plots: int = 50,
    corr_enabled: bool = True,
    vif_enabled: bool = True,
    raw_data_check_enabled: bool = False,
    stress_enabled: bool = False,
    cluster_enabled: bool = False,
    shap_enabled: bool = False,
    
    # Check parameters
    shap_bg_size: int = 100,
    shap_test_size: int = 50,
    
    # Split configuration
    split_strategy: str = "holdout",
    test_size: float = 0.2,
    seed_global: int = 42,
    component_seeds: Optional[Dict[str, Optional[int]]] = None,
    
    # Features
    in_scope_cols: Optional[List[str]] = None,
    
    # Additional options
    **kwargs,
) -> Dict[str, Any]:
    """
    Build a comprehensive rule configuration dictionary.
    
    Args:
        saved_raw: Path to raw data file
        artifacts_dir: Output directory for artifacts
        auc_min: Minimum acceptable AUC
        f1_min: Minimum acceptable F1 score
        ks_min: Minimum acceptable KS statistic
        eda_enabled: Enable EDA check
        eda_max_plots: Maximum EDA plots to generate
        corr_enabled: Enable correlation check
        vif_enabled: Enable VIF check
        raw_data_check_enabled: Enable raw data check
        stress_enabled: Enable stress test
        cluster_enabled: Enable cluster coverage check
        shap_enabled: Enable SHAP analysis
        shap_bg_size: SHAP background sample size
        shap_test_size: SHAP test sample size
        split_strategy: "holdout" or "cv"
        test_size: Test split ratio
        seed_global: Global random seed
        component_seeds: Per-component random seeds
        in_scope_cols: Features to include
        **kwargs: Additional configuration
        
    Returns:
        Configuration dictionary for ValidationEngine
    """
    component_seeds = component_seeds or {}
    
    config = {
        # Paths
        "paths": {
            "raw_data": str(saved_raw) if saved_raw else None,
            "artifacts": str(artifacts_dir),
        },
        
        # Thresholds
        "thresholds": {
            "auc_min": auc_min,
            "f1_min": f1_min,
            "ks_min": ks_min,
        },
        
        # EDA configuration
        "EDACheck": {
            "enabled": eda_enabled,
            "max_plots": eda_max_plots,
            "output_dir": str(artifacts_dir / "eda"),
        },
        
        # Correlation configuration
        "CorrelationCheck": {
            "enabled": corr_enabled,
            "threshold": 0.8,
            "method": "pearson",
            "output_dir": str(artifacts_dir / "correlation"),
        },
        
        # VIF configuration
        "VIFCheck": {
            "enabled": vif_enabled,
            "threshold": 5.0,
            "output_dir": str(artifacts_dir / "vif"),
        },
        
        # Raw data configuration
        "RawDataCheck": {
            "enabled": raw_data_check_enabled,
            "raw_data": str(saved_raw) if saved_raw else None,
        },
        
        # Stress test configuration
        "StressTestCheck": {
            "enabled": stress_enabled,
            "noise_levels": [0.01, 0.05, 0.1],
            "seed": component_seeds.get("stress"),
            "output_dir": str(artifacts_dir / "stress"),
        },
        
        # Input cluster configuration
        "InputClusterCheck": {
            "enabled": cluster_enabled,
            "n_clusters": 10,
            "seed": component_seeds.get("cluster"),
            "output_dir": str(artifacts_dir / "cluster"),
        },
        
        # SHAP configuration
        "SHAPCheck": {
            "enabled": shap_enabled,
            "background_size": shap_bg_size,
            "test_size": shap_test_size,
            "seed": component_seeds.get("shap"),
            "output_dir": str(artifacts_dir / "shap"),
        },
        
        # Split configuration
        "split": {
            "strategy": split_strategy,
            "test_size": test_size,
            "seed": component_seeds.get("split", seed_global),
        },
        
        # Seed
        "seed": seed_global,
        
        # Features
        "in_scope_cols": in_scope_cols or [],
    }
    
    # Merge any additional kwargs
    config.update(kwargs)
    
    return config


# Legacy alias
_build_rule_cfg = build_rule_config
