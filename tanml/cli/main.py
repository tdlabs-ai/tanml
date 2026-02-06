# tanml/cli/main.py
from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys


def _parse_args(argv):
    p = argparse.ArgumentParser(prog="tanml ui", add_help=False)
    p.add_argument("--public", action="store_true", help="Bind on 0.0.0.0 for LAN access")
    p.add_argument("--headless", action="store_true", help="Run without opening a browser")
    p.add_argument("--port", type=int, help="Port to serve on (default 8501)")
    p.add_argument("--max-mb", type=int, help="Max upload/message size in MB (default 2048)")

    p.add_argument("--address", type=str, help="Explicit bind address (overrides --public)")
    p.add_argument("-h", "--help", action="store_true", help="Show help")
    args, _ = p.parse_known_args(argv)
    if args.help:
        p.print_help()
        sys.exit(0)
    return args


def _env_bool(name, default=False):
    v = os.environ.get(name)
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "on"}


def _module_file(modname: str) -> str:
    """Return module file path WITHOUT importing it (avoids early st.* calls)."""
    spec = importlib.util.find_spec(modname)
    if spec is None or not spec.origin:
        print(f"Could not locate module: {modname}", file=sys.stderr)
        sys.exit(1)
    return os.path.abspath(spec.origin)


def _launch_ui(argv):
    # ---- Resolve app path WITHOUT importing tanml.ui.app ----
    app_path = _module_file("tanml.ui.app")

    # ---- Resolve config: CLI > ENV > defaults ----
    args = _parse_args(argv)

    default_address = "127.0.0.1"
    env_address = os.environ.get("TANML_SERVER_ADDRESS")
    address = args.address or ("0.0.0.0" if args.public else (env_address or default_address))

    default_headless = _env_bool("TANML_HEADLESS", False)
    headless = args.headless or default_headless

    default_port = int(os.environ.get("TANML_PORT", "8501"))
    port = args.port if args.port is not None else default_port

    default_max_mb = int(os.environ.get("TANML_MAX_MB", "2048"))
    max_mb = args.max_mb if args.max_mb is not None else default_max_mb

    # ---- Environment for the child process (the Streamlit runner)
    env = os.environ.copy()
    # Marker to indicate app was launched via CLI (for validation in app.py)
    env["TANML_CLI_LAUNCH"] = "1"
    env.setdefault("STREAMLIT_SERVER_MAX_UPLOAD_SIZE", str(max_mb))
    env.setdefault("STREAMLIT_SERVER_MAX_MESSAGE_SIZE", str(max_mb))
    # Skip Streamlit's first-run welcome/email prompt
    env.setdefault("STREAMLIT_CREDENTIALS_EMAIL", "")
    # ALWAYS disable telemetry - we never want Streamlit collecting usage stats
    env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"

    # Optional: reduce auto-reruns in production (cuts stale-media churn)
    env.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")

    # ---- Hand off to the official runner (prevents ScriptRunContext warnings)
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        app_path,
        f"--server.port={port}",
        f"--server.address={address}",
    ]
    # Keep flags too (these mirror the env, fine to be redundant)
    cmd += [
        f"--server.maxUploadSize={max_mb}",
        f"--server.maxMessageSize={max_mb}",
    ]
    if headless:
        cmd.append("--server.headless=true")
    # ALWAYS disable telemetry
    cmd.append("--browser.gatherUsageStats=false")

    return subprocess.call(cmd, env=env)


def main():
    argv = sys.argv[1:]

    # Show help
    if argv and argv[0] in {"-h", "--help"}:
        print(
            "TanML - Industrial-Grade Model Validation Framework\n\n"
            "Usage:\n"
            "  tanml                        Launch the TanML UI\n"
            "  tanml ui [options]           Launch the TanML UI (explicit)\n\n"
            "Options:\n"
            "  --public                     Bind on 0.0.0.0 for LAN access\n"
            "  --headless                   Run without opening a browser\n"
            "  --port N                     Port to serve on (default 8501)\n"
            "  --max-mb N                   Max upload size in MB (default 2048)\n\n"
            "Env vars:\n"
            "  TANML_SERVER_ADDRESS, TANML_HEADLESS, TANML_PORT, TANML_MAX_MB\n\n"
            "Note: Streamlit telemetry is always disabled.\n"
        )
        sys.exit(0)

    # If no args, show help
    if not argv:
        print(
            "TanML - Industrial-Grade Model Validation Framework\n\n"
            "Usage:\n"
            "  tanml ui [options]           Launch the TanML UI\n\n"
            "Options:\n"
            "  --public                     Bind on 0.0.0.0 for LAN access\n"
            "  --headless                   Run without opening a browser\n"
            "  --port N                     Port to serve on (default 8501)\n"
            "  --max-mb N                   Max upload size in MB (default 2048)\n\n"
            "Env vars:\n"
            "  TANML_SERVER_ADDRESS, TANML_HEADLESS, TANML_PORT, TANML_MAX_MB\n\n"
            "Note: Streamlit telemetry is always disabled.\n"
        )
        sys.exit(0)

    # Only "tanml ui" launches the UI
    if argv[0] == "ui":
        ui_args = argv[1:]
        sys.exit(_launch_ui(ui_args))

    # Unknown command
    print(f"Unknown command: {argv[0]}\nTry: tanml --help")
    sys.exit(2)


if __name__ == "__main__":
    main()
