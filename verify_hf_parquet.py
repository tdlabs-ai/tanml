import pandas as pd
from tanml.checks.eda import EDACheck
from tanml.utils.data_loader import load_dataframe
from pathlib import Path
import shutil

# Setup for Rotten Tomatoes Parquet
data_path = "datasets/rotten_tomatoes.parquet"
output_dir = "reports/verify_hf_parquet"
if Path(output_dir).exists():
    shutil.rmtree(output_dir)

print(f"Loading {data_path}...")
df = load_dataframe(data_path)
print(f"Loaded DataFrame shape: {df.shape}")

# Target: 'label' (Classification: neg/pos)
y = df['label']

print("Initializing EDACheck for Parquet Data...")
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
assert metrics['task_type'] == 'classification'
assert metrics['n_rows'] > 1000 # Should be around 8530
assert 'imbalance_score' in metrics

print("✅ Hugging Face Parquet Verification Passed!")
