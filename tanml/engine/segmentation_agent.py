from __future__ import annotations

import os
import joblib
import importlib
from tanml.utils.sas_loader import SASLogisticModel
from tanml.utils.r_loader import RLogisticModel
from tanml.engine.validation_agent import SegmentValidator
from tanml.utils.data_loader import load_dataframe


def handle_segmentation(segment_config, rule_config, args=None, report_output=None):
    global_raw_path = rule_config.get("paths", {}).get("raw_data")
    segment_col = segment_config.get("column")  # ✅ only required for Scenario C
    report_template = report_output  # e.g., "reports/report_{segment}.docx"

    print("🔍 Detected segmentation setup in rules.yaml. Running each segment run separately...")

    for name, run_cfg in segment_config["runs"].items():
        print(f"\n🔹 Validating segment: {name}")
        model_path = run_cfg["model"]

        # CASE 1: Retrain model from cleaned data (model: train)
        if isinstance(model_path, str) and model_path.lower() == "train":
            print(f"🛠️ Retraining model from cleaned data for segment: {name}")

            if "cleaned" in run_cfg:
                cleaned_df_run = load_dataframe(run_cfg["cleaned"])
            else:
                full_cleaned = load_dataframe(rule_config["paths"]["cleaned_data"])
                if segment_col:
                    cleaned_df_run = full_cleaned[full_cleaned[segment_col] == name]
                else:
                    raise ValueError("Missing segment column for slicing cleaned data.")

            X = cleaned_df_run[rule_config["model"]["features"]]
            y = cleaned_df_run[rule_config["model"]["target"]]

            model_source = rule_config.get("model_source", {})
            model_type = model_source.get("type")
            model_module = model_source.get("module")
            model_params = model_source.get("hyperparameters", {})

            if not model_type or not model_module:
                raise ValueError("❌ 'model_source.type' and 'model_source.module' must be defined for retraining.")

            model_class = getattr(importlib.import_module(model_module), model_type)
            model = model_class(**model_params)
            print(f"📦 Using model: {model}")

            model.fit(X, y)
            print(f"✅ Retrained {model_type} for segment '{name}'")

        elif isinstance(model_path, str) and "r_logistic" in model_path.lower():
            model = RLogisticModel(model_path)

        elif isinstance(model_path, str) and model_path.endswith(".pkl"):
            model = joblib.load(model_path)

        elif isinstance(model_path, str) and model_path.endswith(".csv"):
            base = os.path.splitext(model_path)[0]
            model = SASLogisticModel(
                coeffs_path=model_path,
                intercept_path=base + "_intercept.txt",
                feature_order_path=base + "_features.txt"
            )

        else:
            raise ValueError(f"❌ Unsupported model format for segment '{name}': {model_path}")

        # Load and optionally slice raw data
        raw_df_run = None
        if global_raw_path and os.path.exists(global_raw_path):
            full_raw = load_dataframe(global_raw_path)
            if segment_col:
                raw_df_run = full_raw[full_raw[segment_col] == name]
            else:
                raw_df_run = full_raw

        print(f"[DEBUG] raw_df_run for {name}: {type(raw_df_run)}, rows = {len(raw_df_run) if raw_df_run is not None else 'None'}")

        # Load cleaned data
        if "cleaned" in run_cfg:
            cleaned_df_run = load_dataframe(run_cfg["cleaned"])
        else:
            full_cleaned = load_dataframe(rule_config["paths"]["cleaned_data"])
            if segment_col:
                cleaned_df_run = full_cleaned[full_cleaned[segment_col] == name]
            else:
                cleaned_df_run = full_cleaned

        # Format output path
        report_base = report_template.format(segment=name)

        # Run validation
        validator = SegmentValidator(
            segment_column=segment_col,
            segment_values=[name],
            model=model,
            raw_df=raw_df_run,
            cleaned_df=cleaned_df_run,
            target_col=rule_config.get("model", {}).get("target"),
            config=rule_config,
            segment_name=name,
            report_base=report_base
        )
        results = validator.run()

        # Extract report path from SegmentValidator result
        report_base_path = report_template.format(segment=name)
        report_path = os.path.join(report_base_path, f"report_{name}.docx")

        print(f"📄 Report saved: {report_path}")
        print(f"✅ Segment '{name}' validated.")


    print("✅ All segment runs completed.")
    return True
