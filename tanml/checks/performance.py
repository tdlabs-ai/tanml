from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
from scipy.stats import ks_2samp
from .base import BaseCheck
import numpy as np

class PerformanceCheck(BaseCheck):
    def __init__(self, model, X_train, X_test, y_train, y_test, rule_config, cleaned_data):
        super().__init__(model, X_train, X_test, y_train, y_test, rule_config, cleaned_data)
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.rule_config = rule_config or {}

    def run(self):
        """Compute metrics from model predictions on provided test set."""
        result = {
            "accuracy":          self.compute_accuracy(),
            "auc_roc":           self.compute_auc(),
            "f1":                self.compute_f1(),
            "ks":                self.compute_ks(),
            "confusion_matrix":  self.compute_confusion(),
        }
        result["auc"] = result["auc_roc"]  # alias for backward compatibility
        return result

    def compute_accuracy(self):
        y_pred = self.model.predict(self.X_test)
        return round(accuracy_score(self.y_test, y_pred), 4)

    def compute_auc(self):
        y_prob = self.model.predict_proba(self.X_test)[:, 1]
        return round(roc_auc_score(self.y_test, y_prob), 4)

    def compute_f1(self):
        y_pred = self.model.predict(self.X_test)
        return round(f1_score(self.y_test, y_pred), 4)

    def compute_ks(self):
        y_prob = self.model.predict_proba(self.X_test)[:, 1]
        y_true = self.y_test

        # Split by class
        prob_0 = y_prob[y_true == 0]
        prob_1 = y_prob[y_true == 1]

        if len(prob_0) < 2 or len(prob_1) < 2:
            return "Insufficient data for KS test"

        return round(ks_2samp(prob_0, prob_1).statistic, 4)

    def compute_confusion(self):
        y_pred = self.model.predict(self.X_test)
        return confusion_matrix(self.y_test, y_pred).tolist()

    @staticmethod
    def from_predictions(y_true, y_proba):
        """
        Compute metrics directly from true labels and predicted probs.
        Returns a dict: accuracy, auc, f1, ks, confusion_matrix.
        """
        y_true_arr = np.array(y_true)
        y_proba_arr = np.array(y_proba)
        y_pred = (y_proba_arr >= 0.5).astype(int)

        # Avoid invalid metrics when only one class present
        if len(np.unique(y_true_arr)) < 2:
            return {
                "accuracy": "N/A", "auc": "N/A", "f1": "N/A",
                "ks": "N/A", "confusion_matrix": []
            }

        result = {
            "accuracy":         round(accuracy_score(y_true_arr, y_pred), 4),
            "auc_roc":          round(roc_auc_score(y_true_arr, y_proba_arr), 4),
            "f1":               round(f1_score(y_true_arr, y_pred), 4),
            "confusion_matrix": confusion_matrix(y_true_arr, y_pred).tolist(),
        }

        prob_0 = y_proba_arr[y_true_arr == 0]
        prob_1 = y_proba_arr[y_true_arr == 1]

        result["ks"] = (
            round(ks_2samp(prob_0, prob_1).statistic, 4)
            if len(prob_0) >= 2 and len(prob_1) >= 2 else "N/A"
        )

        result["auc"] = result["auc_roc"]  # alias for template/report
        return result
