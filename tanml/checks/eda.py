from .base import BaseCheck
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import scipy.stats as stats

class EDACheck(BaseCheck):
    def __init__(
        self,
        cleaned_data: pd.DataFrame,
        model=None,
        X_train=None,
        X_test=None,
        y_train=None,
        y_test=None,
        rule_config: dict | None = None,
        output_dir: str = "reports/eda",
    ):
        """
        Extensive Exploratory Data Analysis (EDA) check.
        
        Includes:
        - Target Analysis (Balance/Distribution)
        - Global Structure (PCA)
        - Multivariate Outlier Analysis (Isolation Forest)
        - Feature vs Target Relationships
        - Data Quality (Nullity Matrix, Duplicates)
        """
        super().__init__(model, X_train, X_test, y_train, y_test, rule_config, cleaned_data)
        self.cleaned_data = cleaned_data
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.max_plots = (
            rule_config.get("EDACheck", {}).get("max_plots", -1)
            if rule_config is not None else -1
        )
        
        # Determine Task Type if possible (reuse BaseCheck logic or heuristic)
        self.task_type = "classification"
        try:
            if y_train is not None:
                if pd.api.types.is_numeric_dtype(y_train) and y_train.nunique() > 20:
                    self.task_type = "regression"
        except Exception:
            pass

    def run(self):
        print(f"📊 DEBUG — max_plots from YAML = {self.max_plots}")
        artifacts = {}
        
        # 1. Macro Health Check
        summary = self.cleaned_data.describe(include='all').T.fillna('')
        missing = self.cleaned_data.isnull().mean().round(3)
        n_unique = self.cleaned_data.nunique()
        duplicates = self.cleaned_data.duplicated().sum()
        
        # 3. Target Series extraction (moved up for health metrics)
        target_series = None
        if self.y_train is not None:
             target_series = pd.concat([self.y_train, self.y_test]) if self.y_test is not None else self.y_train

        # Health Metrics for Executive Summary
        n_rows, n_cols = self.cleaned_data.shape
        missing_pct_total = (self.cleaned_data.isnull().sum().sum() / (n_rows * n_cols)) if (n_rows * n_cols) > 0 else 0.0
        dup_pct = (duplicates / n_rows) if n_rows > 0 else 0.0
        
        imbalance_score = 0.0
        if self.task_type == "classification" and target_series is not None:
             # Simple score: 1.0 - (min_class / max_class)
             counts = target_series.value_counts()
             if len(counts) > 0:
                 imbalance_score = 1.0 - (counts.min() / counts.max())

        health_metrics = {
            "n_rows": int(n_rows),
            "n_cols": int(n_cols),
            "pct_missing": float(missing_pct_total),
            "pct_duplicates": float(dup_pct),
            "imbalance_score": float(imbalance_score),
            "task_type": self.task_type
        }
        artifacts["health_metrics"] = health_metrics

        summary_path = os.path.join(self.output_dir, "summary_stats.csv")
        missing_path = os.path.join(self.output_dir, "missing_values.csv")
        summary.to_csv(summary_path)
        pd.DataFrame({
            "missing_ratio": missing,
            "n_unique": n_unique
        }).to_csv(missing_path)
        
        artifacts["summary_stats"] = summary_path
        artifacts["missing_values"] = missing_path
        artifacts["duplicate_count"] = int(duplicates)
        
        # 2. Nullity Matrix
        plt.figure(figsize=(10, 6))
        sns.heatmap(self.cleaned_data.isnull(), cbar=False, cmap='viridis')
        plt.title(f"Nullity Matrix (Missing Values Map)")
        nullity_path = os.path.join(self.output_dir, "nullity_matrix.png")
        plt.savefig(nullity_path)
        plt.close()
        artifacts.setdefault("plots", {})["nullity_matrix"] = nullity_path

        # 3. Target Analysis (using pre-calculated target_series)
        
        if target_series is not None:
            plt.figure(figsize=(8, 5))
            if self.task_type == "regression":
                sns.histplot(target_series, kde=True)
                plt.title("Target Distribution (Regression)")
            else:
                # Classification
                counts = target_series.value_counts(normalize=True)
                sns.barplot(x=counts.index, y=counts.values)
                plt.title("Class Balance (Target)")
                plt.ylabel("Proportion")
            
            target_path = os.path.join(self.output_dir, "target_analysis.png")
            plt.savefig(target_path)
            plt.close()
            artifacts["plots"]["target_analysis"] = target_path

        # 4. Global Structure (PCA)
        num_cols = self.cleaned_data.select_dtypes(include="number").columns
        if len(num_cols) >= 2:
            try:
                # Handle missing values for PCA (simple fill)
                X_pca = self.cleaned_data[num_cols].fillna(self.cleaned_data[num_cols].median())
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_pca)
                
                pca = PCA(n_components=2)
                X_proj = pca.fit_transform(X_scaled)
                
                plt.figure(figsize=(8, 6))
                hue = target_series if (target_series is not None and len(target_series) == len(X_proj)) else None
                sns.scatterplot(x=X_proj[:,0], y=X_proj[:,1], hue=hue, alpha=0.7, palette="viridis")
                plt.title(f"PCA 2D Projection (Explained Variance: {pca.explained_variance_ratio_.sum():.2%})")
                plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
                plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
                
                pca_path = os.path.join(self.output_dir, "pca_projection.png")
                plt.savefig(pca_path)
                plt.close()
                artifacts["plots"]["pca_2d"] = pca_path
            except Exception as e:
                print(f"⚠️ PCA failed: {e}")

        # 5. Outlier Analysis (Multivariate - Isolation Forest)
        if len(num_cols) >= 2:
            try:
                # Reuse X_scaled from PCA step if available, else recompute
                X_iso = self.cleaned_data[num_cols].fillna(self.cleaned_data[num_cols].median())
                clf = IsolationForest(max_samples='auto', random_state=42, contamination='auto')
                clf.fit(X_iso)
                scores = clf.decision_function(X_iso)
                
                plt.figure(figsize=(8, 4))
                sns.histplot(scores, kde=True, color='red')
                plt.title("Multivariate Anomaly Scores (Isolation Forest)")
                plt.xlabel("Score (Lower = More Anomalous)")
                iso_path = os.path.join(self.output_dir, "outlier_scores.png")
                plt.savefig(iso_path)
                plt.close()
                artifacts["plots"]["outlier_multivariate"] = iso_path
            except Exception as e:
                print(f"⚠️ Isolation Forest failed: {e}")

        # 6. Feature vs Target (Top Features)
        # Simple heuristic: select top 3 numeric cols by variance if no correlation available
        # In a real engine, we'd use correlation results, but here we keep it independent
        if target_series is not None and len(num_cols) > 0:
            top_feats = self.cleaned_data[num_cols].var().sort_values(ascending=False).head(3).index
            
            ft_plots = []
            for feat in top_feats:
                plt.figure(figsize=(6, 4))
                df_plot = pd.DataFrame({"Feature": self.cleaned_data[feat], "Target": target_series})
                # align indices if possible? Assuming simple concat works if configured correctly
                # Data matching is tricky if cleaned_data was processed differently.
                # Here we assume row alignment for Demo purposes.
                if len(df_plot) > 1000: 
                     df_plot = df_plot.sample(1000, random_state=42)
                
                if self.task_type == "regression":
                     sns.regplot(data=df_plot, x="Feature", y="Target", scatter_kws={'alpha':0.5})
                     plt.title(f"{feat} vs Target")
                else:
                     sns.boxplot(data=df_plot, x="Target", y="Feature")
                     plt.title(f"{feat} by Class")
                
                ft_path = os.path.join(self.output_dir, f"rel_{feat}.png")
                plt.savefig(ft_path)
                plt.close()
                ft_plots.append(ft_path)
            
            artifacts["visualizations"] = ft_plots # Keep logic for generic UI list

        # 7. Basic Univariate (Boxplots + Histograms)
        cols_to_plot = num_cols if self.max_plots in (-1, None) else num_cols[:self.max_plots]
        dist_plots = []
        for col in cols_to_plot:
            # Combo Plot: Histogram + Boxplot
            fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)}, figsize=(7, 5))
            sns.boxplot(x=self.cleaned_data[col], ax=ax_box, orient='h')
            sns.histplot(self.cleaned_data[col], ax=ax_hist, kde=True)
            ax_box.set(yticks=[])
            sns.despine(ax=ax_hist)
            sns.despine(ax=ax_box, left=True)
            plt.title(f"Dist & Outliers: {col}", y=1.3) # move title up
            
            dist_path = os.path.join(self.output_dir, f"dist_{col}.png")
            plt.savefig(dist_path)
            plt.close()
            dist_plots.append(dist_path)
        
        # Merge basic plots into visualizations list for UI
        artifacts.setdefault("visualizations", []).extend(dist_plots)
        
        return artifacts
