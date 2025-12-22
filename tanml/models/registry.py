from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple, Literal

Task = Literal["classification", "regression"]

@dataclass(frozen=True)
class ModelSpec:
    task: Task
    import_path: str                      # e.g., "sklearn.ensemble.RandomForestClassifier"
    defaults: Dict[str, Any] = field(default_factory=dict)
    # UI schema: param -> (type, choices_or_None, help_or_None)
    ui_schema: Dict[str, Tuple[str, Optional[Tuple[Any, ...]], Optional[str]]] = field(default_factory=dict)
    aliases: Dict[str, str] = field(default_factory=dict)  # optional param alias map

# -------------------- 20 MODELS --------------------

_REGISTRY: Dict[Tuple[str, str], ModelSpec] = {
    # -------- Classification (10) --------
    ("sklearn", "LogisticRegression"): ModelSpec(
        task="classification",
        import_path="sklearn.linear_model.LogisticRegression",
        defaults=dict(penalty="l2", solver="lbfgs", C=1.0, class_weight=None, max_iter=1000, random_state=42),
        ui_schema={
            "penalty": ("choice", ("l2", "l1"), "Regularization"),
            "solver": ("choice", ("lbfgs", "liblinear", "saga"), "Solver"),
            "C": ("float", None, "Inverse regularization strength"),
            "class_weight": ("choice", (None, "balanced"), "Imbalance handling"),
            "max_iter": ("int", None, "Max iterations"),
            "random_state": ("int", None, "Seed"),
        },
    ),
    ("sklearn", "RandomForestClassifier"): ModelSpec(
        task="classification",
        import_path="sklearn.ensemble.RandomForestClassifier",
        defaults=dict(n_estimators=400, max_depth=16, min_samples_split=2, min_samples_leaf=1,
                      class_weight=None, random_state=42, n_jobs=-1),
        ui_schema={
            "n_estimators": ("int", None, "Number of trees"),
            "max_depth": ("int", None, "Tree depth (None=unbounded)"),
            "min_samples_split": ("int", None, "Min samples to split"),
            "min_samples_leaf": ("int", None, "Min samples per leaf"),
            "class_weight": ("choice", (None, "balanced", "balanced_subsample"), "Imbalance"),
            "random_state": ("int", None, "Seed"),
        },
    ),
    ("xgboost", "XGBClassifier"): ModelSpec(
        task="classification",
        import_path="xgboost.XGBClassifier",
        defaults=dict(n_estimators=600, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                      min_child_weight=1, reg_lambda=1.0, tree_method="hist", random_state=42, n_jobs=-1),
        ui_schema={
            "n_estimators": ("int", None, "Boosting rounds"),
            "max_depth": ("int", None, "Tree depth"),
            "learning_rate": ("float", None, "Eta"),
            "subsample": ("float", None, "Row subsample"),
            "colsample_bytree": ("float", None, "Column subsample"),
            "min_child_weight": ("float", None, "Min child weight"),
            "reg_lambda": ("float", None, "L2 regularization"),
            "tree_method": ("choice", ("hist", "auto"), "Grow method"),
            "random_state": ("int", None, "Seed"),
            "n_jobs": ("int", None, "Threads"),
        },
    ),
    ("lightgbm", "LGBMClassifier"): ModelSpec(
        task="classification",
        import_path="lightgbm.LGBMClassifier",
        defaults=dict(n_estimators=800, num_leaves=31, max_depth=-1, learning_rate=0.05, subsample=0.8,
                      colsample_bytree=0.8, min_child_samples=20, reg_lambda=1.0, random_state=42, n_jobs=-1),
        ui_schema={
            "n_estimators": ("int", None, "Boosting rounds"),
            "num_leaves": ("int", None, "Max leaves"),
            "max_depth": ("int", None, "-1 = auto"),
            "learning_rate": ("float", None, "Shrinkage"),
            "subsample": ("float", None, "Row subsample"),
            "colsample_bytree": ("float", None, "Column subsample"),
            "min_child_samples": ("int", None, "Min child samples"),
            "reg_lambda": ("float", None, "L2 reg"),
            "random_state": ("int", None, "Seed"),
        },
    ),
    ("sklearn", "SVC"): ModelSpec(
        task="classification",
        import_path="sklearn.svm.SVC",
        defaults=dict(C=1.0, gamma="scale", kernel="rbf", probability=True, class_weight=None, random_state=42),
        ui_schema={
            "C": ("float", None, "Regularization"),
            "gamma": ("choice", ("scale", "auto"), "RBF width"),
            "kernel": ("choice", ("rbf", "linear", "poly", "sigmoid"), "Kernel"),
            "class_weight": ("choice", (None, "balanced"), "Imbalance"),
            "probability": ("bool", None, "Calibrated probs"),
            "random_state": ("int", None, "Seed"),
        },
    ),
    ("sklearn", "KNeighborsClassifier"): ModelSpec(
        task="classification",
        import_path="sklearn.neighbors.KNeighborsClassifier",
        defaults=dict(n_neighbors=15, weights="distance", p=2),
        ui_schema={
            "n_neighbors": ("int", None, "k"),
            "weights": ("choice", ("uniform", "distance"), "Weights"),
            "p": ("int", None, "Minkowski p"),
        },
    ),
    ("sklearn", "GaussianNB"): ModelSpec(
        task="classification",
        import_path="sklearn.naive_bayes.GaussianNB",
        defaults=dict(var_smoothing=1e-9),
        ui_schema={"var_smoothing": ("float", None, "Variance smoothing")},
    ),
    ("catboost", "CatBoostClassifier"): ModelSpec(
        task="classification",
        import_path="catboost.CatBoostClassifier",
        defaults=dict(iterations=800, depth=6, learning_rate=0.05, l2_leaf_reg=3.0, subsample=0.8,
                      loss_function="Logloss", random_state=42, verbose=False),
        ui_schema={
            "iterations": ("int", None, "Rounds"),
            "depth": ("int", None, "Depth"),
            "learning_rate": ("float", None, "Eta"),
            "l2_leaf_reg": ("float", None, "L2 reg"),
            "subsample": ("float", None, "Row subsample"),
            "random_state": ("int", None, "Seed"),
        },
    ),
    ("sklearn", "ExtraTreesClassifier"): ModelSpec(
        task="classification",
        import_path="sklearn.ensemble.ExtraTreesClassifier",
        defaults=dict(n_estimators=400, max_depth=None, min_samples_split=2, min_samples_leaf=1,
                      class_weight=None, random_state=42, n_jobs=-1),
        ui_schema={
            "n_estimators": ("int", None, "Trees"),
            "max_depth": ("int", None, "Depth"),
            "min_samples_split": ("int", None, "Min split"),
            "min_samples_leaf": ("int", None, "Min leaf"),
            "class_weight": ("choice", (None, "balanced", "balanced_subsample"), "Imbalance"),
            "random_state": ("int", None, "Seed"),
        },
    ),
    ("sklearn", "HistGradientBoostingClassifier"): ModelSpec(
        task="classification",
        import_path="sklearn.ensemble.HistGradientBoostingClassifier",
        defaults=dict(max_depth=None, learning_rate=0.1, max_bins=255, l2_regularization=0.0,
                      early_stopping=True, random_state=42),
        ui_schema={
            "max_depth": ("int", None, "Depth (None=auto)"),
            "learning_rate": ("float", None, "Eta"),
            "max_bins": ("int", None, "Bins"),
            "l2_regularization": ("float", None, "L2 reg"),
            "early_stopping": ("bool", None, "Early stop"),
            "random_state": ("int", None, "Seed"),
        },
    ),

    # -------- Regression (10) --------
    ("sklearn", "LinearRegression"): ModelSpec(
        task="regression",
        import_path="sklearn.linear_model.LinearRegression",
        defaults=dict(fit_intercept=True, positive=False),
        ui_schema={
            "fit_intercept": ("bool", None, "Fit intercept"),
            "positive": ("bool", None, "Positive coef"),
        },
    ),
    ("sklearn", "RandomForestRegressor"): ModelSpec(
        task="regression",
        import_path="sklearn.ensemble.RandomForestRegressor",
        defaults=dict(n_estimators=400, max_depth=16, min_samples_split=2, min_samples_leaf=1,
                      random_state=42, n_jobs=-1),
        ui_schema={
            "n_estimators": ("int", None, "Trees"),
            "max_depth": ("int", None, "Depth"),
            "min_samples_split": ("int", None, "Min split"),
            "min_samples_leaf": ("int", None, "Min leaf"),
            "random_state": ("int", None, "Seed"),
        },
    ),
    ("xgboost", "XGBRegressor"): ModelSpec(
        task="regression",
        import_path="xgboost.XGBRegressor",
        defaults=dict(n_estimators=800, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                      min_child_weight=1, reg_lambda=1.0, tree_method="hist", random_state=42, n_jobs=-1),
        ui_schema={
            "n_estimators": ("int", None, "Rounds"),
            "max_depth": ("int", None, "Depth"),
            "learning_rate": ("float", None, "Eta"),
            "subsample": ("float", None, "Row subsample"),
            "colsample_bytree": ("float", None, "Column subsample"),
            "min_child_weight": ("float", None, "Min child weight"),
            "reg_lambda": ("float", None, "L2 reg"),
            "tree_method": ("choice", ("hist", "auto"), "Grow method"),
            "random_state": ("int", None, "Seed"),
            "n_jobs": ("int", None, "Threads"),
        },
    ),
    ("lightgbm", "LGBMRegressor"): ModelSpec(
        task="regression",
        import_path="lightgbm.LGBMRegressor",
        defaults=dict(n_estimators=1200, num_leaves=31, max_depth=-1, learning_rate=0.05, subsample=0.8,
                      colsample_bytree=0.8, min_child_samples=20, reg_lambda=1.0, random_state=42, n_jobs=-1),
        ui_schema={
            "n_estimators": ("int", None, "Rounds"),
            "num_leaves": ("int", None, "Max leaves"),
            "max_depth": ("int", None, "-1 = auto"),
            "learning_rate": ("float", None, "Eta"),
            "subsample": ("float", None, "Row subsample"),
            "colsample_bytree": ("float", None, "Column subsample"),
            "min_child_samples": ("int", None, "Min child samples"),
            "reg_lambda": ("float", None, "L2 reg"),
            "random_state": ("int", None, "Seed"),
        },
    ),
    ("sklearn", "SVR"): ModelSpec(
        task="regression",
        import_path="sklearn.svm.SVR",
        defaults=dict(C=1.0, gamma="scale", epsilon=0.1, kernel="rbf"),
        ui_schema={
            "C": ("float", None, "Regularization"),
            "gamma": ("choice", ("scale", "auto"), "RBF width"),
            "epsilon": ("float", None, "Epsilon tube"),
            "kernel": ("choice", ("rbf", "linear", "poly", "sigmoid"), "Kernel"),
        },
    ),
    ("sklearn", "KNeighborsRegressor"): ModelSpec(
        task="regression",
        import_path="sklearn.neighbors.KNeighborsRegressor",
        defaults=dict(n_neighbors=15, weights="distance", p=2),
        ui_schema={
            "n_neighbors": ("int", None, "k"),
            "weights": ("choice", ("uniform", "distance"), "Weights"),
            "p": ("int", None, "Minkowski p"),
        },
    ),
    ("sklearn", "ElasticNet"): ModelSpec(
        task="regression",
        import_path="sklearn.linear_model.ElasticNet",
        defaults=dict(alpha=0.001, l1_ratio=0.5, max_iter=1000, random_state=42),
        ui_schema={
            "alpha": ("float", None, "Reg strength"),
            "l1_ratio": ("float", None, "L1 vs L2 mix"),
            "max_iter": ("int", None, "Max iterations"),
            "random_state": ("int", None, "Seed"),
        },
    ),
    ("catboost", "CatBoostRegressor"): ModelSpec(
        task="regression",
        import_path="catboost.CatBoostRegressor",
        defaults=dict(iterations=1000, depth=6, learning_rate=0.05, l2_leaf_reg=3.0, subsample=0.8,
                      loss_function="RMSE", random_state=42, verbose=False),
        ui_schema={
            "iterations": ("int", None, "Rounds"),
            "depth": ("int", None, "Depth"),
            "learning_rate": ("float", None, "Eta"),
            "l2_leaf_reg": ("float", None, "L2 reg"),
            "subsample": ("float", None, "Row subsample"),
            "random_state": ("int", None, "Seed"),
        },
    ),
    ("sklearn", "ExtraTreesRegressor"): ModelSpec(
        task="regression",
        import_path="sklearn.ensemble.ExtraTreesRegressor",
        defaults=dict(n_estimators=400, max_depth=None, min_samples_split=2, min_samples_leaf=1,
                      random_state=42, n_jobs=-1),
        ui_schema={
            "n_estimators": ("int", None, "Trees"),
            "max_depth": ("int", None, "Depth"),
            "min_samples_split": ("int", None, "Min split"),
            "min_samples_leaf": ("int", None, "Min leaf"),
            "random_state": ("int", None, "Seed"),
        },
    ),
    ("sklearn", "HistGradientBoostingRegressor"): ModelSpec(
        task="regression",
        import_path="sklearn.ensemble.HistGradientBoostingRegressor",
        defaults=dict(max_depth=None, learning_rate=0.1, max_bins=255, l2_regularization=0.0,
                      early_stopping=True, random_state=42),
        ui_schema={
            "max_depth": ("int", None, "Depth (None=auto)"),
            "learning_rate": ("float", None, "Eta"),
            "max_bins": ("int", None, "Bins"),
            "l2_regularization": ("float", None, "L2 reg"),
            "early_stopping": ("bool", None, "Early stop"),
            "random_state": ("int", None, "Seed"),
        },
    ),
}

# --------------- Helpers ---------------

def list_models(task: Optional[Task] = None) -> Dict[Tuple[str, str], ModelSpec]:
    if task:
        return {k: v for k, v in _REGISTRY.items() if v.task == task}
    return dict(_REGISTRY)

def get_spec(library: str, algo: str) -> ModelSpec:
    key = (library, algo)
    if key not in _REGISTRY:
        raise KeyError(f"Unknown model: {library}.{algo}")
    return _REGISTRY[key]

def _lazy_import(import_path: str) -> Callable[..., Any]:
    mod_name, cls_name = import_path.rsplit(".", 1)
    mod = __import__(mod_name, fromlist=[cls_name])
    return getattr(mod, cls_name)

def build_estimator(library: str, algo: str, params: Optional[Dict[str, Any]] = None):
    spec = get_spec(library, algo)
    Cls = _lazy_import(spec.import_path)
    kwargs = dict(spec.defaults)
    if params:
        canon = {}
        for k, v in params.items():
            k2 = spec.aliases.get(k, k)
            canon[k2] = v
        kwargs.update({k: v for k, v in canon.items() if v is not None})
    return Cls(**kwargs)

def ui_schema_for(library: str, algo: str) -> Dict[str, Tuple[str, Optional[Tuple[Any, ...]], Optional[str]]]:
    return get_spec(library, algo).ui_schema

def infer_task_from_target(y) -> Task:
    import pandas as pd
    import numpy as np
    
    # 1. Float check: if float and has decimals -> Regression
    # (Checking if it's effectively integer, e.g. 1.0, 2.0 is still maybe classification)
    is_float = False
    try:
        # Try to convert to numeric if object
        y_num = pd.to_numeric(y, errors='coerce')
        
        if pd.api.types.is_float_dtype(y_num):
            is_float = True
            # Check if any non-integer values exist
            # dropna is important because nan % 1 is nan
            valid = y_num.dropna()
            if len(valid) > 0 and not np.allclose(valid % 1, 0):
                return "regression"
    except Exception:
        pass

    # 2. Unique count check
    try:
        n = int(y.nunique())  # pandas Series fast path
    except Exception:
        try:
            n = len(set(y))
        except Exception:
            n = 10
            
    # Default rule: few unique values = classification
    # Bumped threshold slightly to 5 to be safer, but float check above handles most regression cases.
    return "classification" if n <= 5 else "regression"
