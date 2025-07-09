import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run TanML model validation toolkit")
    subparsers = parser.add_subparsers(dest="command", required=True)


    validate_parser = subparsers.add_parser("validate", help="Run validation checks and generate report")
    validate_parser.add_argument("--model", required=False,
                        help="Model path: .pkl for sklearn/xgb, .csv for SAS or R logistic")
    validate_parser.add_argument("--raw", required=False, help="Path to raw input data file")
    validate_parser.add_argument("--cleaned", required=False, help="Path to cleaned input data file")
    validate_parser.add_argument("--rules", required=True, help="Path to rules.yaml config file")
    validate_parser.add_argument("--target", required=False, help="Target column name (optional)")
    validate_parser.add_argument("--features", required=False, help="Comma-separated list of features")
    validate_parser.add_argument(
        "--report_path",
        type=str,
        default="reports/final_report.docx",
        help="Path to output DOCX report. Example: --report_path my_reports/run1.docx"
    )


    init_parser = subparsers.add_parser("init", help="Generate starter rules.yaml file")
    init_parser.add_argument("--scenario", required=True, choices=["A", "B", "C"],
                             help="Choose validation scenario: A (single model), B (multiple segments), C (single dataset + segment column)")
    init_parser.add_argument(
        "--output", type=str, default="rules.yaml",
        help="Destination path for rules YAML file (default: rules.yaml)"
    )
    return parser.parse_args()
