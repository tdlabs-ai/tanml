import pytest
import numpy as np
import pandas as pd
from tanml.models.registry import (
    list_models,
    build_estimator,
    infer_task_from_target,
    _REGISTRY
)

# --- list_models ---
def test_list_models_all():
    models = list_models()
    assert len(models) == len(_REGISTRY)
    assert ("sklearn", "LogisticRegression") in models
    assert ("xgboost", "XGBClassifier") in models

def test_list_models_filtered():
    clf_models = list_models(task="classification")
    reg_models = list_models(task="regression")
    
    assert len(clf_models) + len(reg_models) == len(_REGISTRY)
    for m in clf_models.values():
        assert m.task == "classification"
    for m in reg_models.values():
        assert m.task == "regression"

# --- build_estimator ---
def test_build_estimator_sklearn():
    model = build_estimator("sklearn", "LogisticRegression", params={"C": 0.5})
    assert model.C == 0.5
    assert model.solver == "lbfgs"  # from defaults

def test_build_estimator_xgboost():
    # Smoke test for import
    pytest.importorskip("xgboost")
    model = build_estimator("xgboost", "XGBClassifier", params={"n_estimators": 10})
    assert model.n_estimators == 10

def test_build_estimator_invalid():
    with pytest.raises(KeyError):
        build_estimator("sklearn", "NonExistentModel")

# --- infer_task_from_target ---
def test_infer_task_regression_float():
    y = pd.Series([1.1, 2.2, 3.3, 4.4, 5.5])
    assert infer_task_from_target(y) == "regression"

def test_infer_task_regression_many_integers():
    # Many unique integers -> regression
    y = pd.Series(range(100))
    assert infer_task_from_target(y) == "regression"

def test_infer_task_classification_few_integers():
    # Few unique integers -> classification
    y = pd.Series([0, 1, 0, 1, 0, 1])
    assert infer_task_from_target(y) == "classification"

def test_infer_task_classification_strings():
    y = pd.Series(["A", "B", "A", "C"])
    assert infer_task_from_target(y) == "classification"
