import pandas as pd
import os
from statsmodels.stats.outliers_influence import variance_inflation_factor

class VIFCheck:
    def __init__(self, model, X_train, X_test, y_train, y_test, rule_config, cleaned_df, output_dir="reports/vif"):
        self.cleaned_df = cleaned_df.select_dtypes(include=['float64', 'int64']).dropna()
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def run(self):
        result = {}
        try:
            if self.cleaned_df.shape[1] < 2:
                return {
                    "vif_table": [],
                    "high_vif_features": [],
                    "status": "Not enough numeric features"
                }

            X = self.cleaned_df.copy()
            X.insert(0, "Intercept", 1)  # Add constant term for VIF

            vif_data = []
            for i in range(X.shape[1]):
                try:
                    vif = variance_inflation_factor(X.values, i)
                except Exception:
                    vif = float("inf")
                vif_data.append({
                    "feature": X.columns[i],
                    "VIF": round(vif, 2)
                })

            high_vif = [row["feature"] for row in vif_data if row["feature"] != "Intercept" and row["VIF"] > 5]

            # Save to CSV
            output_path = os.path.join(self.output_dir, "vif_table.csv")
            pd.DataFrame(vif_data).to_csv(output_path, index=False)

            result["vif_table"] = vif_data
            result["high_vif_features"] = high_vif
            result["csv_path"] = output_path
            result["status"] = "VIF computed successfully"

        except Exception as e:
            result["vif_table"] = []
            result["high_vif_features"] = []
            result["status"] = f"VIFCheck failed: {str(e)}"

        return result
