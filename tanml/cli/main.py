import argparse
from tanml.cli.validate_cmd import run_validate
from tanml.cli.init_cmd import run_init

def main():
    parser = argparse.ArgumentParser(prog="tanml")
    subparsers = parser.add_subparsers(dest="command")

    # tanml validate --rules path.yaml
    validate_parser = subparsers.add_parser("validate", help="Run model validation")
    validate_parser.add_argument("--rules", required=True, help="Path to rules/config YAML")

    # tanml init --scenario B
    init_parser = subparsers.add_parser("init", help="Initialize rules YAML template")
    init_parser.add_argument("--scenario", choices=["A", "B", "C"], required=True, help="Scenario type")
    init_parser.add_argument("--overwrite", action="store_true", help="Overwrite existing rules.yaml if it exists")
    init_parser.add_argument("--output", default="rules.yaml", help="Path where rules.yaml should be saved (default: rules.yaml)")

    args = parser.parse_args()

    if args.command == "validate":
        run_validate(args.rules)
    elif args.command == "init":
           run_init(args.scenario, dest_path=args.output, overwrite=args.overwrite)

    else:
        parser.print_help()
