from .base import BaseCheck

class RuleEngineCheck(BaseCheck):
    def run(self):
        result = {}
        try:
            perf = self.rule_config.get("check_results", {}).get("PerformanceCheck", {})
            rules = self.rule_config.get("rules", {})

            applied_rules = {}

            for metric, conditions in rules.items():
                actual_value = perf.get(metric)

                if actual_value is None:
                    applied_rules[metric] = "❌ Metric not found"
                    continue

                # Attempt to cast to float
                try:
                    actual_value = float(actual_value)
                except (ValueError, TypeError):
                    applied_rules[metric] = f"❌ Invalid value: {actual_value}"
                    continue

                # Evaluate min/max
                rule_messages = []
                passed = True

                if "min" in conditions and actual_value < conditions["min"]:
                    passed = False
                    rule_messages.append(f"{actual_value:.4f} < min {conditions['min']}")
                if "max" in conditions and actual_value > conditions["max"]:
                    passed = False
                    rule_messages.append(f"{actual_value:.4f} > max {conditions['max']}")

                applied_rules[metric] = "✅ Passed" if passed else "❌ " + " | ".join(rule_messages)

            result["rules"] = applied_rules
            result["overall_pass"] = all(v.startswith("✅") for v in applied_rules.values())

        except Exception as e:
            result["error"] = str(e)

        return result
