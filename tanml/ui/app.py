# tanml/ui/app.py
from __future__ import annotations

import os, time, uuid, json, hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Callable, Tuple, List
import os
import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
import io
import matplotlib.pyplot as plt
import seaborn as sns

from docx import Document
from docx.shared import Inches, RGBColor
from docx.oxml.ns import nsdecls
from docx.oxml import parse_xml
import matplotlib.colors as mcolors

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# TanML internals
from tanml.utils.data_loader import load_dataframe
from tanml.engine.core_engine_agent import ValidationEngine
from tanml.report.report_builder import ReportBuilder
from importlib.resources import files

# Model registry (20-model suite)
from tanml.models.registry import (
    list_models, ui_schema_for, build_estimator, infer_task_from_target, get_spec
)
from tanml.ui.pages.profiling import render_data_profiling_hub
from tanml.ui.pages.preprocessing import render_preprocessing_hub
from tanml.ui.pages.ranking import render_feature_ranking_page
from tanml.ui.pages.model_dev import render_model_development_page


from pathlib import Path
from importlib.resources import files  


try:

    _max_mb = int(os.environ.get("TANML_MAX_MB", "1024"))
    st.set_option("server.maxUploadSize", _max_mb)
    st.set_option("server.maxMessageSize", _max_mb)
    st.set_option("browser.gatherUsageStats", False)
except Exception:
    pass

# --- Report Helpers (imported from reports module) ---
from tanml.ui.reports import (
    _choose_report_template,
    _filter_metrics_for_task,
    _generate_dev_report_docx,
    _generate_eval_report_docx,
    _generate_ranking_report_docx,
    _fmt2,
)


from tanml.ui.config import load_css
from tanml.ui.services.session import _session_dir
from tanml.ui.pages.setup import render_setup_page
from tanml.ui.pages.evaluation import render_model_evaluation_page

def main():
    st.set_page_config(page_title="TanML", layout="wide")
    load_css() # Inject styles

    run_dir = _session_dir()
    
    with st.sidebar:
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


