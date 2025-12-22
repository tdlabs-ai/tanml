import pandas as pd
import numpy as np
from tanml.utils.stats import MetricAggregator
from sklearn.model_selection import RepeatedStratifiedKFold, RepeatedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

def test_aggregator():
    print("Testing MetricAggregator...")
    metrics = [
        {"accuracy": 0.80, "f1": 0.5},
        {"accuracy": 0.82, "f1": 0.6},
        {"accuracy": 0.78, "f1": 0.4},
    ]
    agg = MetricAggregator(metrics)
    report = agg.get_formatted_report()
    summary = agg.get_summary()
    
    print("Report:", report)
    print("Summary:", summary)
    
    # Assert mean calculation
    assert np.isclose(summary["accuracy"]["mean"], 0.80)
    assert "0.800 ± 0.020" in report["accuracy"]
    print("✅ Aggregator Passed")

def test_cv_logic():
    print("\nTesting CV Logic on Iris...")
    # Load data simulation
    data = load_iris(as_frame=True)
    X = data.data
    y = data.target
    
    # Config similar to app.py defaults
    cv_folds = 5
    cv_repeats = 2
    seed_global = 42
    
    cv = RepeatedStratifiedKFold(n_splits=cv_folds, n_repeats=cv_repeats, random_state=seed_global)
    model = DecisionTreeClassifier()
    
    scores = []
    count = 0
    for tr, te in cv.split(X, y):
        count += 1
        X_tr, X_te = X.iloc[tr], X.iloc[te]
        y_tr, y_te = y.iloc[tr], y.iloc[te]
        
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        acc = accuracy_score(y_te, pred)
        scores.append({"accuracy": acc})
        
    print(f"Ran {count} folds (Expected {cv_folds * cv_repeats})")
    assert count == cv_folds * cv_repeats
    
    agg = MetricAggregator(scores)
    print("Final CV Report:", agg.get_formatted_report())
    print("✅ CV Loop Passed")

if __name__ == "__main__":
    test_aggregator()
    test_cv_logic()
