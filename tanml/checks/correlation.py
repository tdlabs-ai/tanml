from .base import BaseCheck
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

class CorrelationCheck(BaseCheck):
    def __init__(self, cleaned_data: pd.DataFrame, output_dir: str = "reports/correlation"):
        """
        Computes Pearson and Spearman correlation matrices and saves them to disk,
        along with a heatmap for visualization.
        """
        self.cleaned_data = cleaned_data
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def run(self):
        # Select numeric features only
        numeric_data = self.cleaned_data.select_dtypes(include="number")

        if numeric_data.shape[1] < 2:
            print("⚠️ Not enough numeric features for correlation.")
            return {
                "pearson_csv": None,
                "spearman_csv": None,
                "heatmap_path": None,
                "error": "Not enough numeric features for correlation",
            }

        # Compute correlations
        pearson_corr = numeric_data.corr(method="pearson")
        spearman_corr = numeric_data.corr(method="spearman")

        # Save CSVs
        pearson_path = os.path.join(self.output_dir, "pearson_corr.csv")
        spearman_path = os.path.join(self.output_dir, "spearman_corr.csv")
        pearson_corr.to_csv(pearson_path)
        spearman_corr.to_csv(spearman_path)

        # Create heatmap
        heatmap_path = os.path.join(self.output_dir, "heatmap.png")
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            pearson_corr,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            cbar_kws={"label": "Pearson Coefficient"},
        )
        plt.title("Pearson Correlation Heatmap")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(heatmap_path)
        plt.close()

        return {
            "pearson_csv": pearson_path,
            "spearman_csv": spearman_path,
            "heatmap_path": heatmap_path,
        }
