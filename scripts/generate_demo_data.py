#!/usr/bin/env python3
"""
Generate reproducible demo datasets for TanML.
This script creates a synthetic credit risk dataset with realistic distributions.
"""

import numpy as np
import pandas as pd
from pathlib import Path

def generate_demo_data(seed=42, output_dir="demo_datasets"):
    """
    Generate synthetic credit risk data and save as train/test CSVs.
    """
    np.random.seed(seed)
    n_samples = 1000
    
    print(f"Generating {n_samples} synthetic samples with seed {seed}...")
    
    # 1. Generate Features with realistic financial distributions
    data = {
        # High cardinality numeric
        "income": np.random.normal(75000, 25000, n_samples).round(2),
        "credit_score": np.random.normal(680, 80, n_samples).clip(300, 850).astype(int),
        "age": np.random.normal(42, 12, n_samples).clip(18, 85).astype(int),
        "loan_amount": np.random.normal(20000, 10000, n_samples).clip(1000, 100000).round(2),
        
        # Ratios
        "debt_to_income": np.random.beta(2, 5, n_samples).round(4),
        
        # Durations/Counts
        "employment_months": np.random.poisson(72, n_samples).clip(0, 500),
        "previous_defaults": np.random.choice([0, 1, 2], p=[0.85, 0.12, 0.03], size=n_samples),
        "savings_balance": (np.random.exponential(10000, n_samples)).round(2),
        "years_at_residence": np.random.poisson(6, n_samples).clip(0, 50),
        "open_credit_lines": np.random.poisson(5, n_samples).clip(1, 30),
    }
    
    df = pd.DataFrame(data)
    
    # 2. Generate Target (Probability of default)
    # Higher score, income, employment -> lower prob
    # More defaults, higher DTI -> higher prob
    z = (
        0.5 * (df["debt_to_income"] * 10) 
        + 1.5 * df["previous_defaults"]
        - 0.005 * (df["credit_score"] - 300)
        - 0.00001 * df["income"]
        + 1.0 # Bias
    )
    prob = 1 / (1 + np.exp(-z))
    df["target"] = (np.random.rand(n_samples) < prob).astype(int)
    
    # 3. Split 80/20
    train_df = df.iloc[:800].copy()
    test_df = df.iloc[800:].copy()
    
    # 4. Save to disk
    out_path = Path(output_dir)
    out_path.mkdir(exist_ok=True)
    
    train_file = out_path / "train.csv"
    test_file = out_path / "test.csv"
    
    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)
    
    print(f"Successfully saved:")
    print(f"  - {train_file} ({len(train_df)} rows)")
    print(f"  - {test_file} ({len(test_df)} rows)")

if __name__ == "__main__":
    generate_demo_data()
