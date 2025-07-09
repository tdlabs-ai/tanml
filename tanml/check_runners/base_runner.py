class BaseCheckRunner:
    def __init__(self, *args, **kwargs):
        pass

    def run(self, *args, **kwargs):
        raise NotImplementedError("Each check runner must implement its own run method.")
