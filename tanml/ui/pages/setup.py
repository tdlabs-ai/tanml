# tanml/ui/pages/setup.py
"""
Setup page logic for TanML UI.
"""

from __future__ import annotations

import streamlit as st
from tanml.ui.services.data import _pick_target


def render_setup_page(run_dir):
    st.header("Home")
    with st.expander("**Quick Guide: Where do I go?**", expanded=True):
        st.markdown("""
        **Do you have a specific problem?** Find the right step below:

        | **If you want to...** | **Go to...** |
        | :--- | :--- |
        | **Understand your dataset** (distribs, missingness, checks) | **Data Profiling** |
        | **Impute missing values** or **Encode categoricals** | **Data Preprocessing** |
        | **Re-check data** after cleaning/encoding | **Data Profiling** |
        | **Find strongest features** before modeling | **Feature Power Ranking** |
        | **Train/Build a model** | **Model Development** |
        | **Evaluate performance** (ROC, Metrics, Stress Test, SHAP) | **Model Evaluation** |
        | **Get an audit-ready report** | **Report Generation** |
        """)

    with st.expander("**Support & Community**", expanded=True):
        st.markdown("""
        **If you find TanML useful, we would appreciate your support:**

        **Like our work?** Please give the project a star on GitHub.
        
        **Want to stay connected?** Follow and connect with us on LinkedIn.
        
        **Have suggestions or feature requests?** Please share feedback using the Feedback Form (available on GitHub).
        
        **Want to contribute?** Fork the repository, make your changes, and submit a Pull Request.
        """)

    # Common Logic: Target Selection (Persisted)
    df_preview = st.session_state.get("df_preview")
    if df_preview is None:
        df_preview = st.session_state.get("df_cleaned")
    if df_preview is None:
        df_preview = st.session_state.get("df_train")
    
    if df_preview is not None:
        st.divider()
        st.subheader("Variable Configuration")
        cols = list(df_preview.columns)
        
        # Try to persist selection
        tgt_idx = 0
        current_tgt = st.session_state.get("target_col")
        if current_tgt and current_tgt in cols:
            tgt_idx = cols.index(current_tgt)
        else:
            def_tgt = _pick_target(df_preview)
            if def_tgt in cols: tgt_idx = cols.index(def_tgt)
            
        target = st.selectbox("Target Column", cols, index=tgt_idx)
        st.session_state["target_col"] = target
        
        # Features
        current_feats = st.session_state.get("feature_cols", [])
        default_feats = [c for c in cols if c != target]
        # Filter existing selection
        valid_feats = [c for c in current_feats if c in cols and c != target]
        if not valid_feats: valid_feats = default_feats
        
        features = st.multiselect("Features", [c for c in cols if c != target], default=valid_feats)
        st.session_state["feature_cols"] = features
        
        st.info(f"Configuration saved! Target: **{target}**, Features: **{len(features)}**")
