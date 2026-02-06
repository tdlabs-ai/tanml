# tanml/ui/pages/setup.py
"""
Setup page logic for TanML UI.
"""

from __future__ import annotations

import streamlit as st

from tanml.ui.services.data import _pick_target

# URLs for Support & Community buttons
GITHUB_URL = "https://github.com/tdlabs-ai/tanml"
LINKEDIN_URL_TANMAY = "https://www.linkedin.com/in/tanmay-sah/"
LINKEDIN_URL_DOLLY = "https://www.linkedin.com/in/dollysah/"
FEEDBACK_URL = "https://forms.gle/qyLtEhQKgnZCUanW7"

# Card styling
CARD_STYLE = """
<style>
.workflow-card {
    background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
    border: 1px solid rgba(102, 126, 234, 0.3);
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    transition: all 0.3s ease;
    height: 160px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}
.workflow-card:hover {
    border-color: rgba(102, 126, 234, 0.7);
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.2);
}
.workflow-card h3 {
    margin: 0.5rem 0;
    font-size: 1.1rem;
    color: #667eea;
}
.workflow-card p {
    margin: 0;
    font-size: 0.85rem;
    color: #666;
}
.workflow-card .icon {
    font-size: 2rem;
}
</style>
"""


def render_setup_page(run_dir):
    # Inject card styling
    st.markdown(CARD_STYLE, unsafe_allow_html=True)

    # Welcome Banner
    st.markdown(
        """
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
    ">
        <h1 style="color: white; margin: 0; font-size: 2.2rem;">
            Welcome to TanML
        </h1>
        <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0; font-size: 1.1rem;">
            Accelerated Model Development — from data to deployment.
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Workflow Cards (2 rows of 3)
    st.markdown(
        """
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 0.8rem 1.5rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    ">
        <h3 style="color: white; margin: 0; font-size: 1.3rem;">
            🚀 Workflow
        </h3>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Row 1: Data Profiling, Preprocessing, Feature Ranking
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
        <div class="workflow-card">
            <div class="icon">📊</div>
            <h3>Data Profiling</h3>
            <p>Analyze distribution, quality, and summary statistics.</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
        <div class="workflow-card">
            <div class="icon">🔧</div>
            <h3>Preprocessing</h3>
            <p>Clean, transform, and encode data for modeling.</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
        <div class="workflow-card">
            <div class="icon">📈</div>
            <h3>Feature Ranking</h3>
            <p>Identify most important features for prediction.</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Row 2: Model Development, Model Evaluation, Generate Reports
    col4, col5, col6 = st.columns(3)

    with col4:
        st.markdown(
            """
        <div class="workflow-card">
            <div class="icon">🤖</div>
            <h3>Model Development</h3>
            <p>Train, tune, and compare multiple algorithms.</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col5:
        st.markdown(
            """
        <div class="workflow-card">
            <div class="icon">🎯</div>
            <h3>Model Evaluation</h3>
            <p>Assess performance using metrics and visualizations.</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col6:
        st.markdown(
            """
        <div class="workflow-card">
            <div class="icon">📝</div>
            <h3>Generate Reports</h3>
            <p>Download audit-ready Word reports.</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.divider()

    # Support & Community Section
    st.markdown(
        """
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 0.8rem 1.5rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    ">
        <h3 style="color: white; margin: 0; font-size: 1.3rem;">
            💬 Support & Community
        </h3>
    </div>
    """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.link_button("⭐ GitHub", GITHUB_URL, width="stretch")

    with col2:
        st.link_button("💼 Tanmay", LINKEDIN_URL_TANMAY, width="stretch")

    with col3:
        st.link_button("💼 Dolly", LINKEDIN_URL_DOLLY, width="stretch")

    with col4:
        st.link_button("📝 Feedback", FEEDBACK_URL, width="stretch")

    st.divider()

    # Variable Configuration (Existing Logic)
    df_preview = st.session_state.get("df_preview")
    if df_preview is None:
        df_preview = st.session_state.get("df_cleaned")
    if df_preview is None:
        df_preview = st.session_state.get("df_train")

    if df_preview is not None:
        st.subheader("⚙️ Variable Configuration")
        cols = list(df_preview.columns)

        # Try to persist selection
        tgt_idx = 0
        current_tgt = st.session_state.get("target_col")
        if current_tgt and current_tgt in cols:
            tgt_idx = cols.index(current_tgt)
        else:
            def_tgt = _pick_target(df_preview)
            if def_tgt in cols:
                tgt_idx = cols.index(def_tgt)

        target = st.selectbox("Target Column", cols, index=tgt_idx)
        st.session_state["target_col"] = target

        # Features
        current_feats = st.session_state.get("feature_cols", [])
        default_feats = [c for c in cols if c != target]
        # Filter existing selection
        valid_feats = [c for c in current_feats if c in cols and c != target]
        if not valid_feats:
            valid_feats = default_feats

        features = st.multiselect("Features", [c for c in cols if c != target], default=valid_feats)
        st.session_state["feature_cols"] = features

        st.info(f"Configuration saved! Target: **{target}**, Features: **{len(features)}**")
