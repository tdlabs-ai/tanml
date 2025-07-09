from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np
import pandas as pd

class StressTestCheck:
    def __init__(self, model, X, y, epsilon=0.01, perturb_fraction=0.2):
        self.model = model
        self.X = X.copy()
        self.y = y
        self.epsilon = epsilon
        self.perturb_fraction = perturb_fraction

    def run(self):
        np.random.seed(42)
        results = []

        # Compute baseline metrics
        try:
            base_proba = self.model.predict_proba(self.X)[:, 1]
            base_pred = (base_proba >= 0.5).astype(int)
            base_auc = roc_auc_score(self.y, base_proba)
            base_acc = accuracy_score(self.y, base_pred)
        except Exception as e:
            print(f"⚠️ Error computing baseline metrics: {e}")
            return []

        # Perturb each numeric feature
        for col in self.X.columns:
            if not pd.api.types.is_numeric_dtype(self.X[col]):
                continue  # skip non-numeric features

            try:
                n_perturb = int(self.perturb_fraction * len(self.X))
                idx = np.random.choice(self.X.index, size=n_perturb, replace=False)

                X_perturbed = self.X.copy()
                X_perturbed.loc[idx, col] += self.epsilon

                perturbed_proba = self.model.predict_proba(X_perturbed)[:, 1]
                perturbed_pred = (perturbed_proba >= 0.5).astype(int)

                pert_auc = roc_auc_score(self.y, perturbed_proba)
                pert_acc = accuracy_score(self.y, perturbed_pred)

                results.append({
                    "feature": col,
                    "perturbation": f"±{round(self.epsilon * 100, 2)}%",
                    "accuracy": round(pert_acc, 4),
                    "auc": round(pert_auc, 4),
                    "delta_accuracy": round(pert_acc - base_acc, 4),
                    "delta_auc": round(pert_auc - base_auc, 4),
                })

            except Exception as e:
                results.append({
                    "feature": col,
                    "perturbation": f"±{round(self.epsilon * 100, 2)}%",
                    "accuracy": "error",
                    "auc": "error",
                    "delta_accuracy": f"Error: {e}",
                    "delta_auc": f"Error: {e}",
                })

        return pd.DataFrame(results)
