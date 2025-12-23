# tanml/ui/pages/evaluation/main.py
"""
Main evaluation page - uses tab registry for extensibility.

This file handles:
- Data upload and loading
- Model configuration and training
- Tab orchestration via registry

The actual tab content is in tabs/ folder modules.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
import streamlit as st

from tanml.utils.data_loader import load_dataframe
from tanml.ui.services.data import _save_upload
from tanml.ui.services.session import _update_report_buffer
from tanml.ui.components.forms import render_model_form
from tanml.ui.reports import _generate_eval_report_docx

from tanml.models.registry import (
    list_models, build_estimator, infer_task_from_target
)

# Import tab system
from tanml.ui.views.evaluation.tabs import (
    TabContext, 
    discover_tabs, 
    get_registered_tabs,
)


def render_model_evaluation_page(run_dir):
    """
    Render the Model Evaluation page.
    
    This function handles data upload, model training, and orchestrates
    the tab rendering using the registry system.
    """
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    ">
        <h2 style="color: white; margin: 0;">🎯 Model Evaluation</h2>
    </div>
    """, unsafe_allow_html=True)
    st.caption("Upload independent Training and Testing datasets to strictly evaluate model performance.")

    # === 1. DATA UPLOAD ===
    c_up1, c_up2 = st.columns(2)
    f_train = c_up1.file_uploader(
        "Upload Training Data", 
        type=["csv", "xlsx", "xls", "parquet", "dta", "sav", "sas7bdat"], 
        key="eval_u_train"
    )
    f_test = c_up2.file_uploader(
        "Upload Testing Data", 
        type=["csv", "xlsx", "xls", "parquet", "dta", "sav", "sas7bdat"], 
        key="eval_u_test"
    )
    
    # Persist uploads
    if f_train:
        ext = Path(f_train.name).suffix
        p_train = _save_upload(f_train, run_dir, f"eval_train{ext}")
        if p_train: 
            st.session_state["eval_path_train"] = p_train
    
    if f_test:
        ext = Path(f_test.name).suffix
        p_test = _save_upload(f_test, run_dir, f"eval_test{ext}")
        if p_test: 
            st.session_state["eval_path_test"] = p_test
        
    # Check readiness
    path_tr = st.session_state.get("eval_path_train")
    path_te = st.session_state.get("eval_path_test")
    
    if not path_tr or not path_te:
        st.info("Please upload both Training and Testing datasets to begin configuration.")
        return

    # Load DataFrames
    try:
        df_train = load_dataframe(path_tr)
        df_test = load_dataframe(path_te)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return
        
    st.success(f"Loaded Data: Train {df_train.shape}, Test {df_test.shape}")
    st.divider()

    # === 2. MODEL CONFIGURATION ===
    with st.container(border=True):
        st.subheader("Model Configuration")
        
        c1, c2 = st.columns([1, 2])
        all_cols = list(df_train.columns)
        
        # Target Selection
        curr_target = st.session_state.get("eval_target_col")
        if curr_target not in all_cols: 
            curr_target = all_cols[-1]
        
        target = c1.selectbox("Target Column", all_cols, index=all_cols.index(curr_target), key="eval_target")
        st.session_state["eval_target_col"] = target
        
        # Verify Target in Test
        if target not in df_test.columns:
            st.error(f"Target '{target}' not found in Testing Data!")
            return

        # Task Inference
        task_type = infer_task_from_target(df_train[target])
        c1.markdown(f'<span class="task-badge">⚙️ Auto-detected Task: {task_type.title()}</span>', unsafe_allow_html=True)
        
        # Feature Selection
        possible_feats = [c for c in all_cols if c != target and c in df_test.columns]
        curr_feats = st.session_state.get("eval_features", [])
        default_feats = [f for f in curr_feats if f in possible_feats]
        if not default_feats: 
            default_feats = possible_feats
        
        features = c2.multiselect("Feature Selection", possible_feats, default=default_feats, key="eval_features_ms")
        st.session_state["eval_features"] = features
        
        if not features:
            st.warning("Select features to train on.")
            return

        # Model Form
        library, algo, hp, task_type = render_model_form(df_train[target], 42, target_name=f"eval_{target}")

    st.divider()

    # === 3. TRAIN MODEL ===
    if st.button("🚀 Evaluate Model", type="primary", width="stretch"):
        with st.spinner("Training model..."):
            X_train = df_train[features]
            y_train = df_train[target]
            X_test = df_test[features]
            y_test = df_test[target]
            
            try:
                model = build_estimator(library, algo, hp)
                model.fit(X_train, y_train)
                st.success("Model Trained successfully!")
                
                y_pred_tr = model.predict(X_train)
                y_pred_te = model.predict(X_test)
                
                # Get probabilities if available
                y_prob_tr = None
                y_prob_te = None
                if hasattr(model, "predict_proba") and task_type == "classification":
                    try:
                        y_prob_tr = model.predict_proba(X_train)[:, 1]
                        y_prob_te = model.predict_proba(X_test)[:, 1]
                    except:
                        pass
                
                # Save to session state
                st.session_state["eval_chk_model"] = model
                st.session_state["eval_chk_data"] = {
                    "X_train": X_train, "y_train": y_train,
                    "X_test": X_test, "y_test": y_test,
                    "features": features, "target": target,
                    "task_type": task_type,
                    "y_pred_tr": y_pred_tr, "y_pred_te": y_pred_te,
                    "y_prob_tr": y_prob_tr, "y_prob_te": y_prob_te,
                }
                # Clear previous results when new model is trained
                st.session_state["eval_context_results"] = {}
                st.session_state["eval_context_images"] = {}
                st.rerun()

            except Exception as e:
                st.error(f"Training Failed: {e}")
                return

    # === 4. RENDER RESULTS (Using Tab Registry) ===
    if "eval_chk_model" not in st.session_state:
        return
    
    model = st.session_state["eval_chk_model"]
    d = st.session_state["eval_chk_data"]
    
    # Build context for tabs
    context = TabContext(
        model=model,
        X_train=d["X_train"],
        X_test=d["X_test"],
        y_train=d["y_train"].values if hasattr(d["y_train"], 'values') else d["y_train"],
        y_test=d["y_test"].values if hasattr(d["y_test"], 'values') else d["y_test"],
        y_pred_train=d["y_pred_tr"],
        y_pred_test=d["y_pred_te"],
        y_prob_train=d.get("y_prob_tr"),
        y_prob_test=d.get("y_prob_te"),
        task_type=d["task_type"],
        features=d["features"],
        results=st.session_state.get("eval_context_results", {}), # Pass session dict by reference
        images=st.session_state.get("eval_context_images", {}),
        target=d["target"],
        run_dir=run_dir,
    )
    
    # Ensure they are in session state (in case of page reload)
    if "eval_context_results" not in st.session_state:
        st.session_state["eval_context_results"] = context.results
    if "eval_context_images" not in st.session_state:
        st.session_state["eval_context_images"] = context.images
    
    # Discover all registered tabs
    discover_tabs()
    tabs = get_registered_tabs()
    
    if not tabs:
        st.warning("No tabs registered. Check tabs/ folder.")
        return
    
    # Create and render tabs
    st.subheader("Results Comparison: Train vs Test")
    
    tab_names = [t.name for t in tabs]
    tab_objects = st.tabs(tab_names)
    
    for tab_obj, tab_def in zip(tab_objects, tabs):
        with tab_obj:
            try:
                tab_def.render_fn(context)
            except Exception as e:
                st.error(f"Error in tab '{tab_def.name}': {e}")
                import traceback
                st.code(traceback.format_exc())
    
    # === 5. REPORT DOWNLOAD ===
    st.divider()
    if st.button("📥 Generate Evaluation Report", key="btn_eval_report"):
        with st.spinner("Generating report..."):
            try:
                # Organize images
                all_imgs = context.images
                
                eval_imgs = {}
                drift_imgs = {}
                cluster_imgs = {}
                benchmark_imgs = {}
                
                for k, v in all_imgs.items():
                    if k == "drift_distribution":
                        drift_imgs["top_distribution"] = v
                    elif k == "cluster_distribution":
                        cluster_imgs["distribution"] = v
                    elif k == "cluster_pca":
                        cluster_imgs["pca_scatter"] = v
                    elif k == "benchmark_comparison":
                        benchmark_imgs["comparison"] = v
                    else:
                        eval_imgs[k] = v

                eval_data = {
                    "evaluation": {
                        "task_type": context.task_type,
                        "metrics_train": context.results.get("metrics_train", {}),
                        "metrics_test": context.results.get("metrics_test", {}),
                        "images": eval_imgs
                    },
                    "drift": context.results.get("drift"),
                    "drift_images": drift_imgs,
                    "stress": context.results.get("stress"),
                    "cluster_coverage": context.results.get("cluster_coverage"),
                    "cluster_images": cluster_imgs,
                    "benchmark": context.results.get("benchmark_results"),
                    "benchmark_images": benchmark_imgs,
                    "explainability": context.results.get("explainability"),
                }
                report_bytes = _generate_eval_report_docx(eval_data)
                
                st.download_button(
                    "⬇️ Download Report",
                    data=report_bytes,
                    file_name=f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M')}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )
                st.success("Report generated!")
            except Exception as e:
                st.error(f"Error generating report: {e}")
