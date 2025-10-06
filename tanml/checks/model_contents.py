# tanml/checks/model_contents.py

import inspect

class ModelContentsCheck:
    def __init__(self, model):
        self.model = model

    def run(self):
        summary = {}

        # 1. Model type and module
        summary["model_class"] = type(self.model).__name__
        summary["module"] = type(self.model).__module__

        # 2. Feature names
        if hasattr(self.model, "feature_names_in_"):
            summary["feature_names_in"] = list(self.model.feature_names_in_)

        # 3. Hyperparameters
        if hasattr(self.model, "get_params"):
            try:
                summary["hyperparameters"] = self.model.get_params()
            except Exception:
                summary["hyperparameters"] = "Could not extract"

        # 4. Coefficients
        if hasattr(self.model, "coef_"):
            try:
                summary["coefficients"] = self.model.coef_.tolist()
            except Exception:
                summary["coefficients"] = "Could not extract"

        # 5. Public attributes
        summary["attributes"] = [
            name for name in dir(self.model)
            if not name.startswith("_") and not inspect.ismethod(getattr(self.model, name))
        ]

        return summary
