# tanml/ui/app.py
from __future__ import annotations

import os
import sys

# ============================================================================
# CLI Launch Guard: Ensure TanML is launched via `tanml ui`, not directly
# ============================================================================
if os.environ.get("TANML_CLI_LAUNCH") != "1":
    print(
        "\n"
        "╔════════════════════════════════════════════════════════════════╗\n"
        "║  ⚠️  TanML must be launched using the CLI command: tanml ui    ║\n"
        "╠════════════════════════════════════════════════════════════════╣\n"
        "║  Running Streamlit directly is not supported.                  ║\n"
        "║                                                                ║\n"
        "║  Usage:                                                        ║\n"
        "║    tanml ui                     # Launch UI                    ║\n"
        "║    tanml ui --port 9000         # Custom port                  ║\n"
        "║    tanml ui --public            # LAN access                   ║\n"
        "║    tanml ui --help              # More options                 ║\n"
        "╚════════════════════════════════════════════════════════════════╝\n"
    )
    sys.exit(1)

import streamlit as st

# TanML Pages
from tanml.ui.views.profiling import render_data_profiling_hub
from tanml.ui.views.preprocessing import render_preprocessing_hub
from tanml.ui.views.ranking import render_feature_ranking_page
from tanml.ui.views.model_dev import render_model_development_page





try:

    _max_mb = int(os.environ.get("TANML_MAX_MB", "1024"))
    st.set_option("server.maxUploadSize", _max_mb)
    st.set_option("server.maxMessageSize", _max_mb)
    st.set_option("browser.gatherUsageStats", False)
except Exception:
    pass




from pathlib import Path

from tanml.ui.config import load_css
from tanml.ui.services.session import _session_dir
from tanml.ui.views.setup import render_setup_page
from tanml.ui.views.evaluation import render_model_evaluation_page

# Logo path
_LOGO_PATH = Path(__file__).parent / "assets" / "logo.png"

def main():
    st.set_page_config(page_title="TanML", layout="wide")
    load_css() # Inject styles

    run_dir = _session_dir()
    
    with st.sidebar:
        if _LOGO_PATH.exists():
            st.image(str(_LOGO_PATH), width="stretch")
        st.title("TanML")
        
        # Sidebar global config
        # st.session_state.setdefault("seed_global", 42) # Keep default logic elsewhere if needed?
        # Moved to Model Validation Advanced Options
        
        nav = st.radio("Navigation", [
            "Home", 
            "Data Profiling", 
            "Data Preprocessing", 
            "Feature Power Ranking", 
            "Model Development", 
            "Model Evaluation"
        ])
        
        st.markdown("---")

    
    if "Home" in nav:
        render_setup_page(run_dir)
    elif "Data Profiling" in nav:
        render_data_profiling_hub(run_dir)
    elif "Data Preprocessing" in nav:
        render_preprocessing_hub(run_dir)
    elif "Feature Power Ranking" in nav:
        render_feature_ranking_page(run_dir)
    elif "Model Development" in nav:
        render_model_development_page(run_dir)
    elif "Model Evaluation" in nav:
        render_model_evaluation_page(run_dir)

if __name__ == "__main__":
    main()


