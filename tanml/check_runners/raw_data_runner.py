# tanml/check_runners/raw_data_runner.py
"""
Raw data check runner.

This check compares raw data statistics against cleaned data
to validate data processing pipeline.
"""

from __future__ import annotations

import os
from typing import Any, Dict

import pandas as pd

from tanml.check_runners.base_runner import BaseCheckRunner
from tanml.checks.raw_data import RawDataCheck
from tanml.utils.data_loader import load_dataframe


class RawDataCheckRunner(BaseCheckRunner):
    """
    Runner for raw vs cleaned data comparison.
    
    Compares the raw input data against cleaned data to document
    the data processing pipeline and validate transformations.
    
    Configuration:
        raw_data: Path to raw data file or DataFrame
        paths.raw_data: Alternative path location
        
    Output:
        - raw_shape: Shape of raw data
        - cleaned_shape: Shape of cleaned data
        - columns_added: Columns added during processing
        - columns_removed: Columns removed during processing
    """
    
    @property
    def name(self) -> str:
        return "RawDataCheck"
    
    def execute(self) -> Dict[str, Any]:
        """
        Compare raw and cleaned data.
        
        Returns:
            Dictionary containing data comparison results
        """
        # Locate raw data from config
        raw_obj = self._get_raw_data()
        
        if raw_obj is None:
            print("ℹ️ RawDataCheck skipped — raw_data not provided in config.")
            return None
        
        # Load if path
        if isinstance(raw_obj, (str, bytes, os.PathLike)):
            raw_obj = load_dataframe(raw_obj)
        
        if not isinstance(raw_obj, pd.DataFrame):
            print("ℹ️ RawDataCheck skipped — raw_data is not a DataFrame or loadable path.")
            return None
        
        # Run the check
        check = RawDataCheck(
            model=self.context.model,
            X_train=self.context.X_train,
            X_test=self.context.X_test,
            y_train=self.context.y_train,
            y_test=self.context.y_test,
            rule_config=self.context.config,
            cleaned_data=self.context.cleaned_df,
            raw_data=raw_obj,
        )
        
        stats = check.run()
        return stats.get("RawDataCheck", stats)
    
    def _get_raw_data(self):
        """Get raw data from various config locations."""
        config = self.context.config
        return (
            config.get("raw_data")
            or (config.get("paths", {}) or {}).get("raw_data")
            or (config.get("paths", {}) or {}).get("raw")
        )


# =============================================================================
# Legacy Compatibility
# =============================================================================

def run_raw_data_check(
    model,
    X_train,
    X_test,
    y_train,
    y_test,
    rule_config: Dict[str, Any],
    cleaned_data: pd.DataFrame,
    *args,
    **kwargs,
) -> Dict[str, Any] | None:
    """Legacy function interface for RawDataCheck."""
    from tanml.core.context import CheckContext
    
    context = CheckContext(
        model=model,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        config=rule_config,
        cleaned_df=cleaned_data,
    )
    
    runner = RawDataCheckRunner(context)
    return runner.run()
