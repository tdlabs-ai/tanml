from __future__ import annotations

from typing import Any, Dict
import pandas as pd

from .base import BaseCheck
from tanml.utils.data_loader import load_dataframe


class RawDataCheck(BaseCheck):
    """
    Raw data sanity metrics. Accepts:
      - DataFrame via `raw_data`, or
      - a file path via `raw_data`, or
      - YAML: paths.raw_data
    """

    def __init__(
        self,
        model,
        X_train,
        X_test,
        y_train,
        y_test,
        rule_config: Dict[str, Any],
        cleaned_data: pd.DataFrame,
        raw_data: Any = None,
    ):
        super().__init__(model, X_train, X_test, y_train, y_test, rule_config, cleaned_data)

        if not hasattr(self, "config") or self.config is None:
            self.config = {}

        if raw_data is not None:
            if isinstance(raw_data, pd.DataFrame):
                self.config["raw_data"] = raw_data
            elif isinstance(raw_data, (str, bytes)):
                self.config["raw_data"] = load_dataframe(raw_data)
            else:
                raise ValueError("raw_data must be a pandas DataFrame or a file path")

    def run(self) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        try:
            df = self.config.get("raw_data")

            # If df is a path-like string, load it now
            if isinstance(df, (str, bytes)):
                df = load_dataframe(df)
                self.config["raw_data"] = df

            # If still missing, try YAML fallbacks
            if not isinstance(df, pd.DataFrame):
                paths = self.config.get("paths") if isinstance(self.config.get("paths"), dict) else None
                raw_path = None
                if paths:
                    raw_path = paths.get("raw_data") or paths.get("raw")
                raw_path = raw_path or self.config.get("raw_data_path") or self.config.get("raw_path")

                if raw_path:
                    df = load_dataframe(raw_path)
                    self.config["raw_data"] = df
                else:
                    raise ValueError("raw_data not found: provide DataFrame or set paths.raw_data in YAML")

            # --- metrics ---
            results["total_rows"] = int(df.shape[0])
            results["total_columns"] = int(df.shape[1])

            miss = df.isnull().mean().round(4)
            results["avg_missing"] = float(miss.mean())
            results["columns_with_missing"] = miss[miss > 0].to_dict()

            results["duplicate_rows"] = int(df.duplicated().sum())

            const_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
            results["constant_columns"] = const_cols

        except Exception as e:
            results["error"] = str(e)

        return {"RawDataCheck": results}
