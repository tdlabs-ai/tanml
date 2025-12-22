import pandas as pd
from tanml.checks.eda import EDACheck
from tanml.utils.data_loader import load_dataframe
from pathlib import Path
import shutil

# Setup for Excel
data_path = "datasets/financial_sample.xlsx"
output_dir = "reports/verify_excel"
if Path(output_dir).exists():
    shutil.rmtree(output_dir)

print(f"Loading {data_path}...")
# This uses openpyxl via pandas
try:
    df = load_dataframe(data_path)
    print(f"Loaded DataFrame shape: {df.shape}")
except ImportError:
    print("❌ openpyxl not installed. Installing...")
    import subprocess
    subprocess.check_call(["pip", "install", "openpyxl"])
    df = load_dataframe(data_path)

# Target: 'Profit' (Regression)
y = df['Profit']

print("Initializing EDACheck for Excel Data...")
check = EDACheck(
    cleaned_data=df,
    y_train=y,
    output_dir=output_dir,
    rule_config={"EDACheck": {"max_plots": 2}}
)

print("Running Check...")
results = check.run()

print("\n--- RESULTS ---")
metrics = results.get("health_metrics", {})
print("Health Metrics:", metrics)

# Assertions
assert metrics['n_rows'] == 700
assert metrics['task_type'] == 'regression'
assert 'pct_missing' in metrics

print("✅ Excel Verification Passed!")
