from __future__ import annotations
import importlib
from datetime import datetime
from pathlib import Path
from importlib.resources import files
from typing import Optional, Dict, Any

import joblib
import pandas as pd
import tzlocal
from sklearn.model_selection import train_test_split

from tanml.cli.arg_parser import parse_args
from tanml.engine.segmentation_agent import handle_segmentation
from tanml.engine.core_engine_agent import ValidationEngine
from tanml.report.report_builder import ReportBuilder
from tanml.utils.data_loader import load_dataframe
from tanml.utils.model_loader import load_model
from tanml.utils.yaml_loader import load_yaml_config


load_yaml = load_yaml_config

def _resolve_template_path(explicit: Optional[str] = None) -> Path:
    """Return a Path to the DOCX report template.

    Priority:
    1.  If the caller supplied an absolute/relative path (`explicit`), use it.
    2.  Fallback to the *packaged* template shipped inside
        `tanml.report.templates`.
    """
    if explicit:
        return Path(explicit).expanduser().resolve()

    # packaged default
    return files("tanml.report.templates").joinpath("report_template.docx")


# ---------------------------------------------------------------------------
# CLI path (Scenario A)
# ---------------------------------------------------------------------------

def validate_from_args() -> None:
    args = parse_args()

    # Read YAML rules if provided via --rules, else an empty dict
    rule_cfg: Dict[str, Any] = load_yaml(args.rules) if args.rules else {}
    explicit_feats = rule_cfg.get("model", {}).get("features", [])

    # Optional feature override from CLI flag
    if not explicit_feats and args.features:
        explicit_feats = [f.strip() for f in args.features.split(",")]
        print(f"✅ Using CLI feature override: {explicit_feats}")

    # -------------------------------------------------------------------
    # Segmentation scenarios (B / C) are handled by a dedicated agent
    # -------------------------------------------------------------------
    if "segment" in rule_cfg and ("runs" in rule_cfg["segment"] or "column" in rule_cfg["segment"]):
        print("✅ Detected segmentation scenario (Scenario B or C).")
        handle_segmentation(rule_cfg["segment"], rule_cfg, args, args.report_path)
        return

    # -------------------------------------------------------------------
    # Scenario A (single model / dataset)
    # -------------------------------------------------------------------
    if not args.model:
        raise ValueError("❌ No model path provided (use --model).")

    model = load_model(args.model)
    raw_df = load_dataframe(args.raw)
    cleaned_df = load_dataframe(args.cleaned)

    # Column sanity check
    if explicit_feats:
        missing = set(explicit_feats) - set(cleaned_df.columns)
        if missing:
            raise ValueError(f"❌ cleaned.csv missing columns: {missing}")

    target = args.target
    if target not in cleaned_df.columns:
        raise ValueError(f"❌ Target column '{target}' not found in cleaned data.")

    y = cleaned_df[target]
    X = cleaned_df.drop(columns=[target])

    # Feature‑name consistency with persisted model
    if hasattr(model, "feature_names_in_"):
        expected = list(model.feature_names_in_)
        if expected != list(X.columns):
            raise ValueError(
                "❌ Feature mismatch\nExpected: "
                f"{expected}\nGot:      {list(X.columns)}"
            )
        print("✅ Feature names match the model expectations.")

    # Train/test split
    test_size = rule_cfg.get("train_test_split", {}).get("test_size", 0.3)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    engine = ValidationEngine(model, X_train, X_test, y_train, y_test, rule_cfg, cleaned_df)
    results = engine.run_all_checks()

    # Meta‑info block for the report
    now = datetime.now(tzlocal.get_localzone())
    results.update(
        {
            "validation_date": now.strftime("%Y-%m-%d %H:%M:%S %Z (UTC%z)"),
            "model_path": args.model,
            "validated_by": "TanML Automated Validator",
            "rules": rule_cfg.get("rules", {}),
        }
    )

    # -------------------------------------------------------------------
    # Report output location
    # -------------------------------------------------------------------
    output_path = Path(args.report_path).expanduser().resolve()
    if output_path.exists() and output_path.is_dir():
        raise ValueError(
            f"'{output_path}' is a directory. Please provide a full .docx filename."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    template_path = _resolve_template_path(rule_cfg.get("output", {}).get("template_path"))
    ReportBuilder(results, template_path=template_path, output_path=output_path).build()
    print(f"\n✅ Report saved to {output_path}")


# ---------------------------------------------------------------------------
# YAML‑driven path (CLI `tanml validate --rules <file>.yaml`)
# ---------------------------------------------------------------------------

def validate_from_yaml(rules_path: str | Path) -> None:
    config = load_yaml_config(rules_path)

    # Segmentation handling first
    report_output_tpl = config.get("output", {}).get("report_path_template", "reports/{segment}")
    if "segment" in config and "runs" in config["segment"]:
        print("🔍 Detected segmentation setup in rules.yaml. Running segment‑wise validation.")
        handle_segmentation(config["segment"], config, args=None, report_output=report_output_tpl)
        return

    # --------------------------
    # Scenario A (single run)
    # --------------------------
    cleaned_df = pd.read_csv(config["paths"]["cleaned_data"])
    print(f"✅ Loaded cleaned data from: {config['paths']['cleaned_data']}")
    print("✅ No segmentation detected. Running single model validation.")

    target = config["model"]["target"]
    features = config["model"]["features"]
    X = cleaned_df[features]
    y = cleaned_df[target]

    # Model loader / trainer
    if config.get("model_source", {}).get("from_pickle", True):
        model_path = config["paths"].get("model")
        if not model_path:
            raise ValueError("❌ 'paths.model' must be provided when using from_pickle = true")
        model = joblib.load(model_path)
        print(f"✅ Loaded model from: {model_path}")
    else:
        model_cfg = config["model_source"]
        module = importlib.import_module(model_cfg["module"])
        model_class = getattr(module, model_cfg["type"])
        model = model_class(**model_cfg.get("hyperparameters", {}))
        model.fit(X, y)
        print(f"✅ Retrained model from: {model_cfg['module']}.{model_cfg['type']}")
        # For SHAP etc.
        if not hasattr(model, "feature_names_in_"):
            model.feature_names_in_ = X.columns.to_numpy()

    # Train/test split
    test_size = config.get("train_test_split", {}).get("test_size", 0.3)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    ctx = {"expected_features": features}
    engine = ValidationEngine(model, X_train, X_test, y_train, y_test, config, cleaned_df, ctx=ctx)
    results = engine.run_all_checks()

    # Meta block
    now = datetime.now(tzlocal.get_localzone())
    results.update(
        {
            "validation_date": now.strftime("%Y-%m-%d %H:%M:%S %Z (UTC%z)"),
            "model_path": config["paths"].get("model", "retrained_from_yaml"),
            "validated_by": "TanML Automated Validator",
            "rules": config.get("rules", {}),
        }
    )

    # Report output
    report_path = Path(config.get("output", {}).get("report_path", "report.docx")).expanduser()
    report_path.parent.mkdir(parents=True, exist_ok=True)

    template_path = _resolve_template_path(config.get("output", {}).get("template_path"))
    ReportBuilder(results, template_path=template_path, output_path=report_path).build()

    print(f"📄 Report saved: {report_path}")
    print("✅ All validations completed.")


if __name__ == "__main__":
    validate_from_args()

