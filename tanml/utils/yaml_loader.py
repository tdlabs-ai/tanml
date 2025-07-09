import yaml

def load_yaml_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)
