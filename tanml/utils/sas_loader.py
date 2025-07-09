# File: tanml/utils/sas_loader.py

import pandas as pd
import numpy as np         


class SASLogisticModel:
    def __init__(self, coeffs_path, intercept_path, feature_order_path):
        self.coeffs_path = coeffs_path
        self.intercept_path = intercept_path
        self.feature_order_path = feature_order_path

        self.coefficients = self._load_coefficients()
        self.intercept = self._load_intercept()
        self.feature_order = self._load_feature_order()

    def _load_coefficients(self):
        return pd.read_csv(self.coeffs_path, index_col=0).squeeze("columns")

    def _load_intercept(self):
        with open(self.intercept_path) as f:
            return float(f.read().strip())

    def _load_feature_order(self):
        with open(self.feature_order_path) as f:
            return [line.strip() for line in f.readlines()]

    def predict_proba(self, X):
        """
        Return a NumPy array shaped (n_samples, 2) like sklearn:
        [:, 0] = P(class 0), [:, 1] = P(class 1)
        """
        X = X[self.feature_order]
        logits = X.dot(self.coefficients) + self.intercept

        # numeric stability clamp
        logits = logits.clip(-700, 700)

        proba_1 = 1 / (1 + np.exp(-logits))
        proba_0 = 1 - proba_1
        return np.vstack([proba_0, proba_1]).T  # shape (n, 2)

    def predict(self, X):
        """
        Return class labels (0/1) based on 0.5 threshold.
        Works with the NumPy array returned by predict_proba().
        """
        proba_1 = self.predict_proba(X)[:, 1]   # probability of class 1
        return (proba_1 >= 0.5).astype(int)

