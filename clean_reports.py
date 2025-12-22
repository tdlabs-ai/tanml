
import os
import shutil
from pathlib import Path

# Clean previous report
run_dir = Path("tanml_runs")
if run_dir.exists():
    shutil.rmtree(run_dir)
os.makedirs("reports", exist_ok=True)
