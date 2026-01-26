# Tutorials

While TanML is primarily a UI-driven tool, its core analytical engine is available as a standard Python library. This allows data scientists to integrate TanML's validation checks into notebooks or automated pipelines.

## Programmatic Usage (Advanced)

### 1. Accessing the Model Registry

You can access the internal factory of pre-configured models directly. This is useful for inspecting default hyperparameters or programmatically selecting algorithms.

```python
from tanml.models.registry import list_models, build_estimator

# List all available regression models
reg_models = list_models(task="regression")
print(f"Available Regression Models: {list(reg_models.keys())}")

# Build a configured XGBoost Classifier
model = build_estimator("xgboost", "XGBClassifier", params={"n_estimators": 500})
print(model)
```

### 2. Calculating Data Drift (PSI)

You can use the `analysis` module to calculate Population Stability Index (PSI) or Kolmogorov-Smirnov (KS) statistics on any two pandas Series, without needing the full UI workflow.

```python
import pandas as pd
import numpy as np
from tanml.analysis.drift import calculate_psi

# Example: Simulate training and production data
train_data = pd.Series(np.random.normal(0, 1, 1000))
prod_data = pd.Series(np.random.normal(0.5, 1, 1000)) # Shifted mean

# Calculate PSI
psi = calculate_psi(train_data, prod_data)

print(f"Population Stability Index: {psi:.4f}")

if psi > 0.2:
    print("🚨 Critical Drift Detected!")
else:
    print("✅ Distribution is stable.")
```

### 3. Check Explained Variance (VIF)

Detect multicollinearity in your feature set programmatically.

```python
from tanml.analysis.correlation import calculate_vif

df = pd.DataFrame({
    "f1": np.random.rand(100),
    "f2": np.random.rand(100)
})
# Create a collinear feature
df["f3"] = df["f1"] * 2 + 0.5 

vif_report = calculate_vif(df)
print(f"High VIF Features: {vif_report['high_vif_features']}")
```
