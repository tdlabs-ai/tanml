
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from tanml.engine.core_engine_agent import ValidationEngine

def run_demo():
    print("Loading data...")
    df = pd.read_csv("datasets/cleaned_a.csv")
    target = "default_flag"
    
    X = df.drop(columns=[target])
    y = df[target]
    
    print(f"Data shape: {df.shape}")
    print(f"Target: {target}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training model...")
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    config = {
        "model": {"type": "classification"},
        "skip_checks": [],
        # Minimal config
        "auc_roc": {"min": 0.5},
        "ks": {"min": 0.0},
        "f1": {"min": 0.0},
    }
    
    print("Running ValidationEngine...")
    engine = ValidationEngine(
        model=model,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        config=config,
        cleaned_data=df
    )
    
    results = engine.run_all_checks()
    
    print("\n--- Validation Results Summary ---")
    summary = results.get("summary", {})
    for k, v in summary.items():
        print(f"{k}: {v}")
        
    if "ClassificationMetrics" in results or "performance" in results:
        print("\nPerformance check passed.")
    else:
        print("\nPerformance check missing?")

if __name__ == "__main__":
    run_demo()
