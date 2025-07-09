"""
ValidationEngine – runs all registered check-runners and assembles a
single results dictionary that the ReportBuilder / Jinja template expects.
"""

from tanml.engine.check_agent_registry import CHECK_RUNNER_REGISTRY
#from tanml.checks.cleaning_repro import CleaningReproCheck 


KEEP_AS_NESTED = {
    "DataQualityCheck",
    "StressTestCheck",
    "InputClusterCheck",
    "RawDataCheck",
    #"CleaningReproCheck",
    "SHAPCheck",
    "VIFCheck",
    "CorrelationCheck",
    "EDACheck",
}


class ValidationEngine:
    def __init__(
        self,
        model,
        X_train,
        X_test,
        y_train,
        y_test,
        config,
        cleaned_data,
        raw_df=None ,
        ctx=None 
    ):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.config = config
        self.cleaned_data = cleaned_data
        self.raw_df = raw_df 

        self.results = dict(config.get("check_results", {}))
        self.ctx = ctx or {}
    def run_all_checks(self):
        for check_name, runner_func in CHECK_RUNNER_REGISTRY.items():
            if check_name in self.config.get("skip_checks", []):
                continue

            print(f"✅ Running {check_name}")
            try:
                result = runner_func(
                    self.model,
                    self.X_train,
                    self.X_test,
                    self.y_train,
                    self.y_test,
                    self.config,
                    self.cleaned_data,
                    raw_df=self.raw_df 
                )

                self._integrate(check_name, result)

            except Exception as e:
                print(f"⚠️  {check_name} failed: {e}")
                self.results[check_name] = {"error": str(e)}

        # Add CleaningReproCheck manually
        # if self.raw_df is not None:
        #     print("✅ Running CleaningReproCheck")
        #     try:
        #         check = CleaningReproCheck(self.raw_df, self.cleaned_data)

        #         self.results["CleaningReproCheck"] = check.run()
        #     except Exception as e:
        #         print(f"⚠️ CleaningReproCheck failed: {e}")
        #         self.results["CleaningReproCheck"] = {"error": str(e)}
        # else:
        #     print("⚠️ Skipping CleaningReproCheck — raw_df not provided")
        #     self.results["CleaningReproCheck"] = {"error": "raw_data not available"}

        # convenience copy for template
        self.results["check_results"] = dict(self.results)
        return self.results

    def _integrate(self, check_name: str, result):
        """Merge a check result into self.results respecting the template layout."""
        if not result:
            return

        # Special flatten for LogisticStatsCheck
        if check_name == "LogisticStatsCheck":
            self.results.update(result)
            return

        # If it's a simple object (rare), store as-is
        if not isinstance(result, dict):
            self.results[check_name] = result
            return

        # Keep entire dict nested
        if check_name in KEEP_AS_NESTED:
            self.results[check_name] = result
            return

        # If runner returns {"CheckName": {...}}, unwrap
        if set(result.keys()) == {check_name}:
            self.results[check_name] = result[check_name]
            return

        # Default: merge into root
        self.results.update(result)
