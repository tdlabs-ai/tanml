# test_tanml_install.py
"""
Test script for TanML v0.1.9.dev1 from TestPyPI.

Setup:
  python3 -m venv .venv
  source .venv/bin/activate
  pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ tanml==0.1.9.dev1
  python test_tanml_install.py
"""

import sys

print("=" * 60)
print("TanML TestPyPI Installation Verification")
print("=" * 60)

# 1. Test basic import
print("\n1. Testing basic import...")
try:
    import tanml
    print(f"   ✅ tanml imported successfully")
except ImportError as e:
    print(f"   ❌ Failed to import tanml: {e}")
    sys.exit(1)

# 2. Test data loader
print("\n2. Testing data loader...")
try:
    from tanml.utils.data_loader import load_dataframe
    print(f"   ✅ load_dataframe imported")
except ImportError as e:
    print(f"   ❌ Failed: {e}")

# 3. Test stress test check
print("\n3. Testing stress test module...")
try:
    from tanml.checks.stress_test import StressTestCheck
    print(f"   ✅ StressTestCheck imported")
except ImportError as e:
    print(f"   ❌ Failed: {e}")

# 4. Test SHAP check (this was the file we fixed)
print("\n4. Testing SHAP check module...")
try:
    from tanml.checks.explainability.shap_check import SHAPCheck
    print(f"   ✅ SHAPCheck imported (syntax error fixed!)")
except SyntaxError as e:
    print(f"   ❌ Syntax Error: {e}")
except ImportError as e:
    print(f"   ⚠️ Import issue (dependency): {e}")

# 5. Test model registry
print("\n5. Testing model registry...")
try:
    from tanml.models.registry import build_estimator
    print(f"   ✅ build_estimator imported")
except ImportError as e:
    print(f"   ❌ Failed: {e}")

# 6. Test CLI entry point
print("\n6. Testing CLI entry point...")
try:
    from tanml.cli.main import main
    print(f"   ✅ CLI main() imported")
except ImportError as e:
    print(f"   ❌ Failed: {e}")

# 7. Quick functional test
print("\n7. Running quick functional test...")
try:
    import pandas as pd
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from tanml.checks.stress_test import StressTestCheck
    
    X = pd.DataFrame({"f1": np.random.rand(50), "f2": np.random.rand(50)})
    y = np.random.randint(0, 2, 50)
    
    model = LogisticRegression()
    model.fit(X, y)
    
    check = StressTestCheck(model, X, y, epsilon=0.1, perturb_fraction=0.2)
    result = check.run()
    
    print(f"   ✅ Stress test ran successfully, result shape: {result.shape}")
except Exception as e:
    print(f"   ❌ Functional test failed: {e}")

print("\n" + "=" * 60)
print("Verification Complete!")
print("=" * 60)
print("\nTo test the UI, run: tanml ui")
