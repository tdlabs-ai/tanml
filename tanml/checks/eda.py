from .base import BaseCheck
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

class EDACheck(BaseCheck):
    def __init__(
        self,
        cleaned_data: pd.DataFrame,
        rule_config: dict | None = None,
        output_dir: str = "reports/eda",
    ):
        """
        Basic Exploratory Data Analysis (EDA) check.

        Generates summary statistics, missing values, and distribution plots.

        Args:
            cleaned_data (pd.DataFrame): Cleaned dataset to analyze.
            rule_config (dict, optional): Rule configuration (YAML) to control plotting limits.
            output_dir (str): Directory where EDA output files will be saved.
        """
        self.cleaned_data = cleaned_data
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # Max number of plots to generate (default = -1 = "all")
        self.max_plots = (
            rule_config.get("EDACheck", {}).get("max_plots", -1)
            if rule_config is not None else -1
        )

    def run(self):
        print(f"📊 DEBUG — max_plots from YAML = {self.max_plots}")

        # 1. Generate summary stats and uniqueness/missing info
        summary = self.cleaned_data.describe(include='all').T.fillna('')
        missing = self.cleaned_data.isnull().mean().round(3)
        n_unique = self.cleaned_data.nunique()

        # 2. Save CSVs
        summary_path = os.path.join(self.output_dir, "summary_stats.csv")
        missing_path = os.path.join(self.output_dir, "missing_values.csv")
        summary.to_csv(summary_path)
        pd.DataFrame({
            "missing_ratio": missing,
            "n_unique": n_unique
        }).to_csv(missing_path)

        # 3. Create histograms for numeric columns
        num_cols = self.cleaned_data.select_dtypes(include="number").columns
        cols_to_plot = num_cols if self.max_plots in (-1, None) else num_cols[:self.max_plots]

        for col in cols_to_plot:
            plt.figure(figsize=(6, 4))
            sns.histplot(self.cleaned_data[col], kde=True)
            plt.title(f"Distribution of {col}")
            plt.savefig(os.path.join(self.output_dir, f"dist_{col}.png"))
            plt.close()

        # 4. Return metadata
        return {
            "summary_stats":  summary_path,
            "missing_values": missing_path,
            "visualizations": [f"dist_{c}.png" for c in cols_to_plot],
        }
