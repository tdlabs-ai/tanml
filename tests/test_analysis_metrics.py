import numpy as np
import pandas as pd

from tanml.analysis.correlation import (
    calculate_correlation_matrix,
    calculate_vif,
    find_highly_correlated_pairs,
)
from tanml.analysis.drift import analyze_drift, calculate_psi


# --- Drift Tests ---
def test_calculate_psi_no_drift():
    # Identical distributions should have PSI ~ 0
    expected = pd.Series(np.random.normal(0, 1, 1000))
    actual = expected.copy()
    psi = calculate_psi(expected, actual)
    assert psi < 0.05
    assert psi >= 0


def test_calculate_psi_moderate_drift():
    # Shifted distribution
    np.random.seed(42)
    expected = pd.Series(np.random.normal(0, 1, 1000))
    actual = pd.Series(np.random.normal(0.5, 1, 1000))
    psi = calculate_psi(expected, actual)
    assert psi > 0.05  # Depending on bins/randomness, usually > 0.1


def test_calculate_psi_empty():
    assert np.isnan(calculate_psi(pd.Series([]), pd.Series([])))


def test_analyze_drift_integration():
    train_df = pd.DataFrame({"f1": np.random.randn(100)})
    test_df = pd.DataFrame({"f1": np.random.randn(100) + 10})  # Huge drift

    results = analyze_drift(train_df, test_df)
    assert "f1" in results
    assert results["f1"]["has_drift"] is True
    assert results["f1"]["drift_level"] == "severe"


# --- Correlation Tests ---
def test_correlation_matrix():
    df = pd.DataFrame(
        {
            "a": [1, 2, 3, 4],
            "b": [2, 4, 6, 8],  # Perfect corr with a
            "c": [5, 4, 3, 2],  # Perfect neg corr with a
        }
    )
    corr = calculate_correlation_matrix(df)
    assert np.isclose(corr.loc["a", "b"], 1.0)
    assert np.isclose(corr.loc["a", "c"], -1.0)


def test_find_highly_correlated_pairs():
    df = pd.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [1, 2, 3],
            "c": [0.1, 0.9, 0.5],  # Uncorrelated
        }
    )
    corr = df.corr()
    pairs = find_highly_correlated_pairs(corr, threshold=0.99)
    assert len(pairs) == 1
    assert set([pairs[0]["feature_1"], pairs[0]["feature_2"]]) == {"a", "b"}


def test_vif_calculation():
    # Multicollinear: b = 2*a
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [2, 4, 6, 8, 10], "c": [5, 4, 1, 6, 2]})
    result = calculate_vif(df)
    features = result["vif_values"]

    # a and b should have high (infinite) VIF due to perfect correlation
    assert features["a"] > 100 or np.isinf(features["a"])
    assert features["b"] > 100 or np.isinf(features["b"])
    assert "a" in result["high_vif_features"]
