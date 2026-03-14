# tanml/ui/pages/setup.py
"""
Setup page logic for TanML UI.
"""

from __future__ import annotations

import streamlit as st

from tanml.ui.services.data import _load_demo_data, _pick_target

# URLs for Support & Community buttons
GITHUB_URL = "https://github.com/tdlabs-ai/tanml"
LINKEDIN_URL_TANMAY = "https://www.linkedin.com/in/tanmay-sah/"
LINKEDIN_URL_DOLLY = "https://www.linkedin.com/in/dollysah/"
FEEDBACK_URL = "https://forms.gle/qyLtEhQKgnZCUanW7"

# Card styling
CARD_STYLE = """
<style>
.card-link {
    text-decoration: none !important;
    color: inherit !important;
    display: block;
    margin-bottom: 1.5rem;
}
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
    cursor: pointer;
}
.card-link:hover .workflow-card {
    border-color: rgba(102, 126, 234, 0.7);
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.2);
    background: linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.15) 100%);
    transform: translateY(-2px);
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

    # Workflow Grid Header
    st.markdown(
        """
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 0.8rem 1.5rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
    ">
        <h3 style="color: white; margin: 0; font-size: 1.3rem;">
            🚀 Workflow
        </h3>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Grid Layout
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
        <a href="/?page=Data+Profiling" target="_self" class="card-link">
            <div class="workflow-card">
                <div class="icon">📊</div>
                <h3>Data Profiling</h3>
                <p>Analyze distribution, quality, and summary statistics.</p>
            </div>
        </a>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
        <a href="/?page=Data+Preprocessing" target="_self" class="card-link">
            <div class="workflow-card">
                <div class="icon">🔧</div>
                <h3>Data Preprocessing</h3>
                <p>Clean, transform, and split data for modeling.</p>
            </div>
        </a>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
        <a href="/?page=Feature+Power+Ranking" target="_self" class="card-link">
            <div class="workflow-card">
                <div class="icon">📈</div>
                <h3>Feature Power Ranking</h3>
                <p>Identify most important features for prediction.</p>
            </div>
        </a>
        """,
            unsafe_allow_html=True,
        )

    col4, col5, col6 = st.columns(3)

    with col4:
        st.markdown(
            """
        <a href="/?page=Model+Development" target="_self" class="card-link">
            <div class="workflow-card">
                <div class="icon">🤖</div>
                <h3>Model Development</h3>
                <p>Train, tune, and compare multiple algorithms.</p>
            </div>
        </a>
        """,
            unsafe_allow_html=True,
        )

    with col5:
        st.markdown(
            """
        <a href="/?page=Model+Evaluation" target="_self" class="card-link">
            <div class="workflow-card">
                <div class="icon">🎯</div>
                <h3>Model Evaluation</h3>
                <p>Assess performance using metrics and visualizations.</p>
            </div>
        </a>
        """,
            unsafe_allow_html=True,
        )

    with col6:
        st.markdown(
            """
        <a href="/?page=Model+Evaluation" target="_self" class="card-link">
            <div class="workflow-card">
                <div class="icon">📝</div>
                <h3>Generate Reports</h3>
                <p>Download audit ready reports.</p>
            </div>
        </a>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 0.8rem 1.5rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
    ">
        <h3 style="color: white; margin: 0; font-size: 1.3rem;">
            📖 TanML Workflow Guide
        </h3>
    </div>
    
    <div style="
        background: rgba(102, 126, 234, 0.05);
        border-left: 4px solid #667eea;
        padding: 1.5rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 2rem;
    ">
        <p style="margin-bottom: 1.2rem;">
            <b>Step 1: Understand Your Data (Data Profiling)</b><br>
            Start by uploading your raw dataset. TanML will automatically analyze your data quality onscreen, identifying missing values, highlighting outliers, and visualizing the distribution of each variable to provide a baseline understanding of your data.
        </p>
        <p style="margin-bottom: 1.2rem;">
            <b>Step 2: Clean & Split (Data Preprocessing)</b><br>
            Raw data often requires preparation before modeling. Upload your dataset to apply imputation strategies for missing values and encode text categories. Then, use the built-in <b>Data Splitter</b> at the bottom of the page to divide your cleaned dataset into a dedicated <i>Training Set</i> (for model learning) and an independent <i>Testing Set</i> (for final validation). You can export and download these datasets for the next steps.
        </p>
        <p style="margin-bottom: 1.2rem;">
            <b>Step 3: Analyze Features (Feature Power Ranking)</b><br>
            Upload your newly created <i>Training Set</i>. TanML will mathematically rank your features onscreen to identify which ones show predictive power for your target variable, and which ones are statistically insignificant. Use these insights to drop low-ranking features and streamline your dataset. <b>Click 'Generate Feature Report' to download a detailed report of your variables.</b>
        </p>
        <p style="margin-bottom: 1.2rem;">
            <b>Step 4: Build & Iterate (Model Development)</b><br>
            Upload your optimized <i>Training Set</i> and experiment with different algorithms, from classical regressions to tree-based models. TanML handles Cross-Validation in the background to ensure your model learns the underlying patterns, helping you identify the best performing configuration. <b>Click 'Generate Development Report' to download a comprehensive summary of your model's performance.</b>
        </p>
        <p style="margin-bottom: 0;">
            <b>Step 5: Validate Performance (Model Evaluation)</b><br>
            Once you have selected a model configuration from Step 4, upload your original <i>Training Set</i> alongside your unseen <i>Testing Set</i>. TanML will evaluate your final model onscreen to check how well it generalizes to unseen data, monitor for Data Drift, and generate Explainability (SHAP) plots to help interpret the results. <b>Click 'Generate Evaluation Report' to download a complete, audit ready document proving your final model's real-world readiness.</b>
        </p>
    </div>
    """, unsafe_allow_html=True)

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
