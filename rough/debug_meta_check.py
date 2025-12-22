
import pandas as pd
import numpy as np
from tanml.checks.model_meta import ModelMetaCheck

# Mock Model
class MockModel:
    def get_params(self): return {}
    def __module__(self): return "sklearn"

model = MockModel()
X = pd.DataFrame(np.random.rand(10, 2))
y = np.random.rand(10) # Continuous float

# TEST 1: Regression Config
print("--- TEST 1: Regression Config ---")
cfg_reg = {"task_type": "regression"}
check = ModelMetaCheck(model, X, X, y, y, cfg_reg, None)
res = check.run()
print("Result keys:", res.keys())
print("Target Balance Type:", type(res["target_balance"]))
print("Target Balance Content:", res["target_balance"])

# TEST 2: Default Config (Should be Value Counts but let's see)
print("\n--- TEST 2: Classification Config (Implicit) ---")
cfg_cls = {} # Defaults to classification
check2 = ModelMetaCheck(model, X, X, y, y, cfg_cls, None)
res2 = check2.run()
# This might look messy because y is floats, but let's see if it differs
print("Target Balance Content (First 5):", str(res2["target_balance"])[:100])
