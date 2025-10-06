from .base import BaseCheck
import pandas as pd

class DataQualityCheck(BaseCheck):
    def __init__(self, model, X_train, X_test, y_train, y_test, rule_config, cleaned_data):
        super().__init__(model, X_train, X_test, y_train, y_test, rule_config, cleaned_data)

    def run(self):
        df = self.cleaned_data
        results = {}

        if not isinstance(df, pd.DataFrame):
            return {"error": "Cleaned data is not a valid DataFrame"}

        # Missing value analysis
        missing_ratio = df.isnull().mean()
        results["avg_missing"] = round(missing_ratio.mean(), 4)
        results["columns_with_missing"] = {
            col: round(val, 4) for col, val in missing_ratio.items() if val > 0
        }

        # Constant columns (same value across all rows)
        constant_cols = [col for col in df.columns if df[col].nunique(dropna=False) == 1]
        results["constant_columns"] = constant_cols

        return results
