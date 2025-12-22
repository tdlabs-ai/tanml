import pandas as pd
from tanml.checks.eda import EDACheck
from pathlib import Path
import shutil

# Setup for Iris
data_path = "datasets/iris.csv"
output_dir = "reports/verify_iris"
if Path(output_dir).exists():
    shutil.rmtree(output_dir)

print(f"Loading {data_path}...")
df = pd.read_csv(data_path)
y = df['species'] # Target is 'species'

print("Initializing EDACheck for Iris...")
check = EDACheck(
    cleaned_data=df,
    y_train=y,
    output_dir=output_dir,
    rule_config={"EDACheck": {"max_plots": 5}}
)

print("Running Check...")
results = check.run()

print("\n--- RESULTS ---")
metrics = results.get("health_metrics", {})
print("Health Metrics:", metrics)

# Assertions
assert metrics['n_rows'] == 150
assert metrics['n_cols'] == 5 # sepal_length, sepal_width, petal_length, petal_width, species
assert metrics['task_type'] == 'classification', f"Got {metrics['task_type']}"
assert metrics['imbalance_score'] == 0.0, "Iris should be perfectly balanced"

print("✅ Iris Verification Passed!")
