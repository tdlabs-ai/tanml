import numpy as np
import pandas as pd
from typing import List, Dict, Any, Union

class MetricAggregator:
    """
    Aggregates a list of metric dictionaries from Cross-Validation folds.
    Computes Mean, Std, Median, IQR, Min, Max for each numeric metric.
    """
    
    def __init__(self, metrics_list: List[Dict[str, Any]]):
        self.metrics_list = metrics_list
        self.df = pd.DataFrame(metrics_list)
        
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Returns a dictionary where keys are metric names and values are 
        stats dictionaries (mean, std, median, etc.)
        """
        summary = {}
        # Filter for numeric columns only
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            vals = self.df[col].dropna()
            if vals.empty:
                continue
                
            summary[col] = {
                "mean": float(vals.mean()),
                "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                "median": float(vals.median()),
                "min": float(vals.min()),
                "max": float(vals.max()),
                "q1": float(vals.quantile(0.25)),
                "q3": float(vals.quantile(0.75)),
                "iqr": float(vals.quantile(0.75) - vals.quantile(0.25)),
                "count": int(len(vals))
            }
        return summary

    def get_formatted_report(self) -> Dict[str, str]:
        """
        Returns string formatted 'Mean ± Std' for UI display.
        """
        summary = self.get_summary()
        report = {}
        for metric, stats in summary.items():
            report[metric] = f"{stats['mean']:.3f} ± {stats['std']:.3f}"
        return report
