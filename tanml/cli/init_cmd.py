from tanml.utils.yaml_generator import generate_rules_yaml

def run_init(scenario, dest_path="rules.yaml", overwrite=False):
    try:
        generate_rules_yaml(scenario=scenario, dest_path=dest_path, overwrite=overwrite)

    except Exception as e:
        print(f"❌ Failed to create YAML: {e}")
