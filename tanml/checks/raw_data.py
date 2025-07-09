# tanml/checks/raw_data.py  ← make sure this is the ONLY copy on disk
from .base import BaseCheck
import pandas as pd

class RawDataCheck(BaseCheck):
    def __init__(self,
                 model, X_train, X_test, y_train, y_test,
                 rule_config, cleaned_data,
                 raw_data=None):
        # bring in rule_config & cleaned_data
        super().__init__(model, X_train, X_test, y_train, y_test,
                         rule_config, cleaned_data)

        if not hasattr(self, "config") or self.config is None:
            self.config = {}

        if raw_data is not None:
            if isinstance(raw_data, (str, bytes)):
                raw_data = pd.read_csv(raw_data)
            if not isinstance(raw_data, pd.DataFrame):
                raise ValueError("raw_data must be a pandas DataFrame or CSV path")
            self.config["raw_data"] = raw_data

    def run(self):
        results = {}
        try:
            df = self.config.get("raw_data")         
            if not isinstance(df, pd.DataFrame):
                raise ValueError("raw_data not found or not a DataFrame")

            results["total_rows"]   = int(df.shape[0])
            results["total_columns"] = int(df.shape[1])

            miss = df.isnull().mean().round(4)
            results["avg_missing"]            = float(miss.mean())
            results["columns_with_missing"]   = miss[miss > 0].to_dict()

            results["duplicate_rows"] = int(df.duplicated().sum())

            const_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
            results["constant_columns"] = const_cols

        except Exception as e:
            results["error"] = str(e)

        return {"RawDataCheck": results}

