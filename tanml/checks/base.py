from abc import ABC, abstractmethod

class BaseCheck(ABC):
    def __init__(self, model, X_train, X_test, y_train, y_test, rule_config, cleaned_data):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.rule_config = rule_config
        self.cleaned_data = cleaned_data



    @abstractmethod
    def run(self):
        """
        This method must be implemented by every check.
        It should return a dictionary of results.
        """
