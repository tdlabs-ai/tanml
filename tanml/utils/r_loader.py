import pandas as pd
import numpy as np

class RLogisticModel:
    """
    Wrapper for logistic regression models exported from R.
    Assumes CSV with columns: ID, y_true, y_pred_proba
    """

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.df = pd.read_csv(model_path)

        # Check required columns exist
        expected_cols = {'y_true', 'y_pred_proba'}
        if not expected_cols.issubset(set(self.df.columns)):
            raise ValueError(f"R model CSV must contain columns: {expected_cols}")

        self.y_true = self.df['y_true'].values
        self.y_pred_proba = self.df['y_pred_proba'].values

    def predict_proba(self, X=None):
        """
        Mimics sklearn’s predict_proba format: n_samples x 2
        """
        proba = self.y_pred_proba.reshape(-1, 1)
        return np.hstack([1 - proba, proba])

    def get_true_labels(self):
        return self.y_true
