from .base import BaseCheck
import pandas as pd

class ModelMetaCheck(BaseCheck):
    def __init__(self, model, X_train, X_test, y_train, y_test, rule_config, cleaned_data):
        super().__init__(model, X_train, X_test, y_train, y_test, rule_config, cleaned_data)

    def run(self):
        result = {}
        try:
            # Basic metadata
            result["model_type"] = type(self.model).__name__
            result["model_class"] = type(self.model).__name__
            result["module"] = getattr(self.model, "__module__", "Unknown")

            # Features
            result["n_features"] = self.X_train.shape[1]
            result["feature_names"] = list(getattr(self.X_train, "columns", []))

            # Training stats
            result["n_train_rows"] = self.X_train.shape[0]
            y_series = pd.Series(self.y_train)
            result["target_balance"] = y_series.value_counts().to_dict()

            # Hyperparameters
            try:
                params = self.model.get_params()
                result["hyperparam_table"] = [
                    {"param": k, "value": str(v)} for k, v in params.items()
                ]
            except Exception as e:
                result["hyperparam_table"] = [{"param": "error", "value": str(e)}]

            # Public attributes
            try:
                result["attributes"] = {
                    k: str(v)
                    for k, v in self.model.__dict__.items()
                    if not k.startswith("_")
                }
            except Exception as e:
                result["attributes"] = {"error": str(e)}

            result["status"] = "Model metadata extracted successfully"

        except Exception as e:
            result["error"] = str(e)
            result["status"] = "ModelMetaCheck failed"

        return result
