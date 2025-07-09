from pathlib import Path
import shutil

def generate_rules_yaml(
    scenario: str = "A",
    dest_path: str = "rules.yaml",
    overwrite: bool = False
):

    scenario = scenario.upper()
    template_map = {
        "A": "rules_one_model_one_dataset.yaml",
        "B": "rules_multiple_models_datasets.yaml",
        "C": "rules_one_dataset_segment_column.yaml"
    }

    if scenario not in template_map:
        raise ValueError("Invalid scenario. Must be 'A', 'B', or 'C'.")

    src = Path(__file__).parent.parent / "config_templates" / template_map[scenario]
    if not src.exists():
        raise FileNotFoundError(f"Template not found at {src}")

    dst = Path(dest_path)
    if dst.exists() and not overwrite:
        raise FileExistsError(f"{dst} already exists. Use --overwrite to replace it.")

    if dst.parent and not dst.parent.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)

    shutil.copyfile(src, dst)

    print(f"✅ Created: {dst.resolve()} for Scenario {scenario}")
    print("👉 Now edit this YAML to fill in your model, data, and feature details.")
