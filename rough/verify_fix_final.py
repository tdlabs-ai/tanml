
import pandas as pd
import numpy as np
from tanml.engine.core_engine_agent import ValidationEngine
from tanml.report.report_builder import ReportBuilder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from tanml.checks.model_meta import ModelMetaCheck

# 1. Load Data
df = pd.read_csv("sample_regression.csv")
X = df.drop(columns="target")
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Model
model = RandomForestRegressor(n_estimators=10, random_state=42)
model.fit(X_train, y_train)

# 3. Config
rule_cfg = {
    "task_type": "regression",
    "model_meta_enabled": True,
    # Inject CV stats to match UI flow (mocked for speed)
    "cv_stats": {
        "rmse": {"mean": 0.5, "std": 0.1, "p05": 0.4, "p50": 0.5, "p95": 0.6},
        "mae": {"mean": 0.4, "std": 0.05, "p05": 0.35, "p50": 0.4, "p95": 0.45},
        "r2": {"mean": 0.9, "std": 0.02, "p05": 0.88, "p50": 0.9, "p95": 0.92},
        "median_ae": {"mean": 0.4, "std": 0.05, "p05": 0.35, "p50": 0.4, "p95": 0.45}
    }
}

# 4. Check Metadata explicitly
print("Checking ModelMetaCheck logic...")
meta_check = ModelMetaCheck(model, X_train, X_test, y_train, y_test, rule_cfg, None)
meta_res = meta_check.run()
print("Target Balance Result:", meta_res["target_balance"])

# 5. Run Validation Engine
print("Running Engine...")
# We need to construct engine context
ctx = {}
engine = ValidationEngine(model, X_train, X_test, y_train, y_test, rule_cfg, pd.DataFrame(), ctx=ctx)
engine.run_all_checks()
results = engine.results
results["summary"]["cv_stats"] = rule_cfg["cv_stats"] # ensure stats are in summary

# 6. Build Report
print("Building Report...")
template_path = "tanml/report/templates/report_template_reg.docx"
out_path = "Final_Regression_Report.docx"

builder = ReportBuilder(results, template_path=template_path, output_path=out_path)
builder.build()
print(f"Generated {out_path}")
