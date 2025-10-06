from tanml.checks.rule_engine import RuleEngineCheck

def RuleEngineCheckRunner(model, X_train, X_test, y_train, y_test, rule_config, cleaned_df, *args, **kwargs):
    check = RuleEngineCheck(model, X_train, X_test, y_train, y_test, rule_config, cleaned_df)
    return check.run()
