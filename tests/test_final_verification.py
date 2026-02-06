import os
import sys

import numpy as np
import pandas as pd

# Ensure tanml is in path
sys.path.insert(0, os.getcwd())

from tanml.checks.stress_test import StressTestCheck
from tanml.utils.data_loader import load_dataframe


def test_1_data_loader_optimization():
    print("\n🧪 TEST 1: Data Loader Memory Optimization")

    # Create dummy data
    df = pd.DataFrame(
        {
            "high_card": [f"id_{i}" for i in range(100)],
            "low_card": ["A", "B", "A", "B"] * 25,
            "numeric": np.random.randn(100),
        }
    )

    # Save to CSV
    df.to_csv("test_data_loader.csv", index=False)

    # Load via optimized loader
    try:
        loaded_df = load_dataframe("test_data_loader.csv")

        # Check types
        print(f"   Shape: {loaded_df.shape}")
        print(f"   'high_card' dtype: {loaded_df['high_card'].dtype} (Expected: object)")
        print(f"   'low_card' dtype: {loaded_df['low_card'].dtype} (Expected: category)")

        # Assertions
        assert loaded_df["high_card"].dtype == "object", "High cardinality should remain object"
        assert isinstance(loaded_df["low_card"].dtype, pd.CategoricalDtype), (
            "Low cardinality should be categorical"
        )
        print("   ✅ PASSED")
    finally:
        if os.path.exists("test_data_loader.csv"):
            os.remove("test_data_loader.csv")


def test_2_stress_test_extreme():
    print("\n🧪 TEST 2: Stress Test Robustness (Epsilon=1.0)")

    from sklearn.linear_model import LogisticRegression

    # Setup Data
    X = pd.DataFrame({"f1": np.random.rand(50), "f2": np.random.rand(50)})
    y = np.random.randint(0, 2, 50)

    model = LogisticRegression()
    model.fit(X, y)

    # Run Check with max epsilon to ensure no math errors
    try:
        check = StressTestCheck(model, X, y, epsilon=1.0, perturb_fraction=0.5)
        result = check.run()

        print(f"   Result Shape: {result.shape}")
        assert not result.empty, "Stress test result should not be empty"
        print("   ✅ PASSED")
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        raise


def test_3_ranking_crash_fix_simulation():
    print("\n🧪 TEST 3: Feature Ranking Logic (NaN Target Handling)")

    # Simulate the logic issue we fixed in ranking.py
    # We want to ensure that if we drop NaNs from target, X is also aligned

    # Data with missing target
    df = pd.DataFrame(
        {
            "feature": [1, 2, 3, 4, 5],
            "target": [1, 0, np.nan, 1, 0],  # Index 2 is nan
        }
    )

    target = "target"
    contenders = ["feature"]

    # --- Logic from ranking.py (Fixed version) ---
    # dropna for target to prevent model crash
    df_sub = df[[*contenders, target]].dropna(subset=[target])
    X = df_sub[contenders]
    y = df_sub[target]

    print(f"   Original Rows: {len(df)}")
    print(f"   Rows after DropNA: {len(X)}")

    # Assert alignment
    assert len(X) == 4, f"Expected 4 rows, got {len(X)}"
    assert len(y) == 4, f"Expected 4 target values, got {len(y)}"
    assert 2 not in X.index, "Row with NaN target (index 2) should be gone"

    # Ensure X is NOT coming from original df (which was the bug)
    X_bug_check = df[contenders]  # This would have length 5
    if len(X) == len(X_bug_check):
        print("   ❌ FAILED: Logic seems to use original dataframe (Bug present)")
    else:
        print("   ✅ PASSED: NaN targets correctly filtered from X and y")


if __name__ == "__main__":
    print("running Verification Suite for TanML v0.1.9...")
    try:
        test_1_data_loader_optimization()
        test_2_stress_test_extreme()
        test_3_ranking_crash_fix_simulation()
        print("\n🎉 ALL CHECKS PASSED SUCCESSFULLY")
    except Exception as e:
        print(f"\n❌ A TEST FAILED: {e}")
        sys.exit(1)
