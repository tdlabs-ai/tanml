# tanml/utils/model_loader.py

import os
import joblib
from tanml.utils.sas_loader import SASLogisticModel
from tanml.utils.r_loader import RLogisticModel

def load_model(model_path):
    """
    Load a model from path. Supports:
    - sklearn/xgboost .pkl
    - SAS .csv with _intercept.txt and _features.txt
    - R exported logistic CSV
    """
    if not model_path:
        raise ValueError("❌ No model path provided.")

    if "r_logistic" in model_path.lower():
        print("✅ Detected R Logistic Regression model")
        return RLogisticModel(model_path)

    elif model_path.endswith(".pkl"):
        print(f"✅ Loading sklearn/XGB model from {model_path}")
        return joblib.load(model_path)

    elif model_path.endswith(".csv"):
        base = os.path.splitext(model_path)[0]
        return SASLogisticModel(
            coeffs_path=model_path,
            intercept_path=base + "_intercept.txt",
            feature_order_path=base + "_features.txt"
        )

    else:
        raise ValueError("❌ Unsupported model format. Use .pkl, .csv, or R model CSV")
