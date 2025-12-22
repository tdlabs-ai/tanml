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
            
            
            task = str(self.rule_config.get("task_type", "classification")).lower().strip()
            
            # Robustly detect if target is numeric/continuous
            # Try to convert to numeric to catch object-dtype columns that are actually floats
            try:
                y_numeric = pd.to_numeric(y_series, errors='coerce')
                # If valid conversion for sufficient data points, use the numeric series
                if y_numeric.notna().sum() > 0:
                     y_check = y_numeric
                else:
                     y_check = y_series
            except:
                y_check = y_series

            is_numeric = pd.api.types.is_numeric_dtype(y_check)
            unique_count = y_check.nunique()
            
            # Use regression summary if explicitly regression OR (numeric and high cardinality)
            if task == "regression" or (is_numeric and unique_count > 10):
                # Continuous distribution summary
                # Use the numeric version for stats calculation to avoid errors
                y_stats = pd.to_numeric(y_series, errors='coerce')
                result["target_balance"] = {
                    "Range": f"{y_stats.min():.2f} to {y_stats.max():.2f}",
                    "Mean": f"{y_stats.mean():.2f}",
                    "Std": f"{y_stats.std():.2f}",
                    "Note": "Continuous target (min/max/mean/std)"
                }
            else:
                # Classification balance - limit to top 50 to avoid massive reports if something goes wrong
                try:
                    counts = y_series.value_counts(normalize=True)
                    if len(counts) > 20: 
                        # Fallback for accident: too many classes
                        result["target_balance"] = {
                            "Top 10 Classes": counts.head(10).to_dict(),
                            "Note": f"Showing top 10 of {len(counts)} unique values"
                        }
                    else:
                        result["target_balance"] = counts.to_dict()
                except Exception as e:
                     result["target_balance"] = {"Error": str(e)}

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
