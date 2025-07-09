from tanml.checks.base import BaseCheck
import shap
import pandas as pd
import matplotlib.pyplot as plt
import traceback
from pathlib import Path
from datetime import datetime


class SHAPCheck(BaseCheck):
    def __init__(self, model, X_train, X_test, y_train, y_test, rule_config=None, cleaned_df=None):
        super().__init__(model, X_train, X_test, y_train, y_test, rule_config, cleaned_data=cleaned_df)
        self.cleaned_df = cleaned_df

    def run(self):
        result = {}

        try:
            expl_cfg = self.rule_config.get("explainability", {})
            bg_n = expl_cfg.get("background_sample_size", 100)
            test_n = expl_cfg.get("test_sample_size", 200)

            X_sample = self.X_test[:test_n]
            background = shap.utils.sample(self.X_train, bg_n, random_state=42)

            X_sample = pd.DataFrame(X_sample)
            background = pd.DataFrame(background)

            explainer = shap.Explainer(self.model, background)
            shap_exp = explainer(X_sample)

            if shap_exp.values.ndim == 3:
                shap_exp.values = shap_exp.values[:, :, 1]
                shap_exp.base_values = shap_exp.base_values[:, 1]

            segment = self.rule_config.get("meta", {}).get("segment", "global")
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = Path(f"reports/images/shap_summary_{segment}_{ts}.png")
            output_path.parent.mkdir(parents=True, exist_ok=True)

            plt.figure(figsize=(8, 6))
            shap.plots.beeswarm(shap_exp, show=False)
            plt.savefig(output_path, bbox_inches="tight")
            plt.close()

            print(f"✅ SHAP plot saved at: {output_path}")
            result["shap_plot_path"] = str(output_path)
            result["status"] = "SHAP plot generated successfully"

        except Exception:
            err = traceback.format_exc()
            print(f"⚠️ SHAPCheck failed:\n{err}")
            result["status"] = f"SHAP plot failed:\n{err}"

        return result
