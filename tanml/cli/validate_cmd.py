# tanml/cli/validate_cmd.py

from tanml.validate import validate_from_yaml

def run_validate(rules_path):
    print(f"🧪 Starting validation using rules from: {rules_path}")
    validate_from_yaml(rules_path)
