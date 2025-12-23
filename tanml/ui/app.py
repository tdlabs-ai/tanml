# tanml/ui/app.py
from __future__ import annotations

import os
import streamlit as st

# TanML Pages
from tanml.ui.pages.profiling import render_data_profiling_hub
from tanml.ui.pages.preprocessing import render_preprocessing_hub
from tanml.ui.pages.ranking import render_feature_ranking_page
from tanml.ui.pages.model_dev import render_model_development_page





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
from tanml.ui.pages.setup import render_setup_page
from tanml.ui.pages.evaluation import render_model_evaluation_page

# Logo path
_LOGO_PATH = Path(__file__).parent / "assets" / "logo.png"

def main():
    st.set_page_config(page_title="TanML", layout="wide")
    load_css() # Inject styles

    run_dir = _session_dir()
    
    with st.sidebar:
        if _LOGO_PATH.exists():
            st.image(str(_LOGO_PATH), use_container_width=True)
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


