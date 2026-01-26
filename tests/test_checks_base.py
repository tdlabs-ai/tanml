import pytest
import pandas as pd
import numpy as np
from tanml.checks.base import BaseCheck, CheckResult, CheckRegistry

# Mock concrete implementation for testing
class MockCheck(BaseCheck):
    name = "Mock Check"
    def run(self) -> CheckResult:
        score = 0.9
        return CheckResult(
            name=self.name,
            status="pass",
            metrics={"score": score}
        )

@CheckRegistry.register
class RegisteredMockCheck(BaseCheck):
    name = "Registered Mock"
    def run(self) -> CheckResult:
        return CheckResult(name=self.name, status="fail")

def test_check_registry_registration():
    # Check if RegisteredMockCheck was auto-registered by decorator
    assert "Registered Mock" in CheckRegistry.list_names()
    cls = CheckRegistry.get("Registered Mock")
    assert cls == RegisteredMockCheck

def test_check_registry_manual_register():
    class ManualCheck(BaseCheck):
        name = "Manual Check"
        def run(self): pass
        
    CheckRegistry.register(ManualCheck)
    assert "Manual Check" in CheckRegistry.list_names()

def test_base_check_initialization():
    X = pd.DataFrame({"a": [1, 2]})
    y = pd.Series([0, 1])
    check = MockCheck(model=None, X_train=X, X_test=X, y_train=y, y_test=y)
    
    assert check.X_train.shape == (2, 1)
    assert check.name == "Mock Check"

def test_check_run_result():
    X = pd.DataFrame({"a": [1]})
    y = pd.Series([0])
    check = MockCheck(None, X, X, y, y)
    result = check.run()
    
    assert isinstance(result, CheckResult)
    assert result.status == "pass"
    assert result.metrics["score"] == 0.9

def test_check_infer_task_type():
    X = pd.DataFrame({"a": [1]})
    
    # Classification (few unique)
    y_cls = pd.Series([0, 1, 0, 1])
    check_cls = MockCheck(None, X, X, y_cls, y_cls)
    assert check_cls._infer_task_type() == "classification"
    
    # Regression (many unique)
    y_reg = pd.Series(np.random.rand(100))
    check_reg = MockCheck(None, X, X, y_reg, y_reg)
    assert check_reg._infer_task_type() == "regression"
