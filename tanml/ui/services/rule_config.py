# tanml/ui/services/rule_config.py
"""
Configuration building service for TanML UI.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional


def _derive_component_seeds(
    global_seed: int, 
    *, 
    split_random: bool,
    stress_enabled: bool, 
    cluster_enabled: bool, 
    shap_enabled: bool
) -> Dict[str, Optional[int]]:
    base = int(global_seed)
    return {
        "split":   base if split_random else None,
        "model":   base + 1,
        "stress":  (base + 2) if stress_enabled else None,
        "cluster": (base + 3) if cluster_enabled else None,
        "shap":    (base + 4) if shap_enabled else None,
    }

def _build_rule_cfg(
    *,
    saved_raw: Optional[Path],
    auc_min: float,
    f1_min: float,
    ks_min: float,
    eda_enabled: bool,
    eda_max_plots: int,
    corr_enabled: bool,
    vif_enabled: bool,
    raw_data_check_enabled: bool,
    model_meta_enabled: bool,
    stress_enabled: bool,
    stress_epsilon: float,
    stress_perturb_fraction: float,
    cluster_enabled: bool,
    cluster_k: int,
    cluster_max_k: int,
    shap_enabled: bool,
    shap_bg_size: int,
    shap_test_size: int,
    artifacts_dir: Path,
    split_strategy: str,
    test_size: float,
    seed_global: int,
    component_seeds: Dict[str, Optional[int]],
    in_scope_cols: List[str],
) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {
        "paths": {},
        "data": {
            "source": "separate" if split_strategy == "supplied" else "single",
            "split": {
                "strategy": split_strategy,
                "test_size": float(test_size),
            },
            "in_scope_columns": list(in_scope_cols),
        },
        "checks_scope": {"use_only_in_scope": True, "reference_split": "train"},
        "options": {"save_artifacts_dir": str(artifacts_dir)},
        "reproducibility": {
            "seed_global": int(seed_global),
            "component_seeds": component_seeds,
        },
        "auc_roc": {"min": float(auc_min)},
        "f1": {"min": float(f1_min)},
        "ks": {"min": float(ks_min)},
        "EDACheck": {"enabled": bool(eda_enabled), "max_plots": int(eda_max_plots)},
        "correlation": {"enabled": bool(corr_enabled)},
        "VIFCheck": {"enabled": bool(vif_enabled)},
        "raw_data_check": {"enabled": bool(raw_data_check_enabled)},
        "model_meta": {"enabled": bool(model_meta_enabled)},
        "StressTestCheck": {
            "enabled": bool(stress_enabled),
            "epsilon": float(stress_epsilon),
            "perturb_fraction": float(stress_perturb_fraction),
        },
        "InputClusterCoverageCheck": {
            "enabled": bool(cluster_enabled),
            "n_clusters": int(cluster_k),
            "max_k": int(cluster_max_k),
        },
        "explainability": {
            "shap": {
                "enabled": bool(shap_enabled),
                "background_sample_size": int(shap_bg_size),
                "test_sample_size": int(shap_test_size),
            }
        },
        "train_test_split": {"test_size": float(test_size)},
    }
    if saved_raw:
        cfg["paths"]["raw_data"] = str(saved_raw)
    return cfg
