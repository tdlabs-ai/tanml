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
from tanml.ui.services.data import _save_upload
from importlib.resources import files

# Model registry (20-model suite)
from tanml.models.registry import (
    list_models, ui_schema_for, build_estimator, infer_task_from_target, get_spec
)


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
from tanml.ui.components.forms import render_model_form
from tanml.ui.services.session import _update_report_buffer



def render_model_evaluation_page(run_dir):
    st.header("Model Evaluation")
    st.caption("Upload independent Training and Testing datasets to strictly evaluate model performance.")

    # 1. Data Upload
    c_up1, c_up2 = st.columns(2)
    f_train = c_up1.file_uploader("Upload Training Data", type=["csv", "xlsx", "xls", "parquet", "dta", "sav", "sas7bdat"], key="eval_u_train")
    f_test = c_up2.file_uploader("Upload Testing Data", type=["csv", "xlsx", "xls", "parquet", "dta", "sav", "sas7bdat"], key="eval_u_test")
    
    # Persist and Load
    if f_train:
        # Preserve original extension
        ext = Path(f_train.name).suffix
        p_train = _save_upload(f_train, run_dir, f"eval_train{ext}")
        if p_train: st.session_state["eval_path_train"] = p_train
    
    if f_test:
        ext = Path(f_test.name).suffix
        p_test = _save_upload(f_test, run_dir, f"eval_test{ext}")
        if p_test: st.session_state["eval_path_test"] = p_test
        
    # Check readiness
    path_tr = st.session_state.get("eval_path_train")
    path_te = st.session_state.get("eval_path_test")
    
    if not path_tr or not path_te:
        st.info("Please upload both Training and Testing datasets to begin configuration.")
        return

    # Load DFs
    try:
        df_train = load_dataframe(path_tr)
        df_test = load_dataframe(path_te)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return
        
    st.success(f"Loaded Data: Train {df_train.shape}, Test {df_test.shape}")
    st.divider()

    # 2. Configuration
    with st.container(border=True):
        st.subheader("Model Configuration")
        
        c1, c2 = st.columns([1, 2])
        all_cols = list(df_train.columns)
        
        # Target Selection
        curr_target = st.session_state.get("eval_target_col")
        if curr_target not in all_cols: curr_target = all_cols[-1]
        
        target = c1.selectbox("Target Column", all_cols, index=all_cols.index(curr_target), key="eval_target")
        st.session_state["eval_target_col"] = target
        
        # Verify Target in Test
        if target not in df_test.columns:
            st.error(f"Target '{target}' not found in Testing Data!")
            return

        # Task Inference
        temp_task = infer_task_from_target(df_train[target])
        c1.markdown(f'<span class="task-badge">⚙️ Auto-detected Task: {temp_task.title()}</span>', unsafe_allow_html=True)
        
        # Feature Selection
        possible_feats = [c for c in all_cols if c != target and c in df_test.columns]
        curr_feats = st.session_state.get("eval_features", [])
        default_feats = [f for f in curr_feats if f in possible_feats]
        if not default_feats: default_feats = possible_feats
        
        features = c2.multiselect("Feature Selection", possible_feats, default=default_feats, key="eval_features_ms")
        st.session_state["eval_features"] = features
        
        if not features:
            st.warning("Select features to train on.")
            return

        # Model Form
        # Helper returns: library, algo, hp, task_type
        # We reuse the global helper but feed it our local Train data sample
        library, algo, hp, task_type = render_model_form(df_train[target], 42, target_name=f"eval_{target}")

    st.divider()

    # 3. Execution (Train & Save)
    if st.button("🚀 Evaluate Model", type="primary", use_container_width=True):
        st.write("Training model...")
        
        # Prepare Data
        X_train = df_train[features]
        y_train = df_train[target]
        X_test = df_test[features]
        y_test = df_test[target]
        
        # Build & Train
        try:
            model = build_estimator(library, algo, hp)
            model.fit(X_train, y_train)
            st.success("Model Trained successfully on Training Data.")
            
            # Predict on Train & Test
            y_pred_tr = model.predict(X_train)
            y_pred_te = model.predict(X_test)
            
            # Helper to calc metrics
            def _calc_metrics(y_true, y_pred, X_in):
                scores_dict = {}
                if task_type == "classification":
                     from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, 
                                                  average_precision_score, brier_score_loss, log_loss, balanced_accuracy_score, matthews_corrcoef)
                     from scipy.stats import ks_2samp
                     
                     scores_dict["accuracy"] = accuracy_score(y_true, y_pred)
                     scores_dict["precision"] = precision_score(y_true, y_pred, zero_division=0)
                     scores_dict["recall"] = recall_score(y_true, y_pred, zero_division=0)
                     scores_dict["f1"] = f1_score(y_true, y_pred, zero_division=0)
                     scores_dict["bal_acc"] = balanced_accuracy_score(y_true, y_pred)
                     scores_dict["mcc"] = matthews_corrcoef(y_true, y_pred)
                     
                     if hasattr(model, "predict_proba"):
                         try:
                             y_prob = model.predict_proba(X_in)[:, 1]
                             scores_dict["roc_auc"] = roc_auc_score(y_true, y_prob)
                             scores_dict["pr_auc"] = average_precision_score(y_true, y_prob)
                             scores_dict["brier"] = brier_score_loss(y_true, y_prob)
                             scores_dict["log_loss"] = log_loss(y_true, y_prob)
                             scores_dict["gini"] = 2 * scores_dict["roc_auc"] - 1
                             
                             p0 = y_prob[y_true==0]
                             p1 = y_prob[y_true==1]
                             scores_dict["ks"] = ks_2samp(p0, p1).statistic
                         except: pass
                else:
                     from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, r2_score
                     scores_dict["r2"] = r2_score(y_true, y_pred)
                     scores_dict["mse"] = mean_squared_error(y_true, y_pred)
                     scores_dict["rmse"] = np.sqrt(scores_dict["mse"])
                     scores_dict["mae"] = mean_absolute_error(y_true, y_pred)
                     scores_dict["median_ae"] = median_absolute_error(y_true, y_pred)
                return scores_dict
            
            scores_tr = _calc_metrics(y_train, y_pred_tr, X_train)
            scores_te = _calc_metrics(y_test, y_pred_te, X_test)
            
            # --- SAVE TO SESSION STATE ---
            st.session_state["eval_chk_model"] = model
            st.session_state["eval_chk_data"] = {
                "X_train": X_train, "y_train": y_train,
                "X_test": X_test, "y_test": y_test,
                "features": features
            }
            st.session_state["eval_chk_results"] = {
                "scores_tr": scores_tr, "scores_te": scores_te,
                "y_pred_tr": y_pred_tr, "y_pred_te": y_pred_te
            }
            st.rerun() # Force rerun to pick up state

        except Exception as e:
            st.error(f"Training Failed: {e}")
            return


    # 4. Results Rendering (Persistent)
    if "eval_chk_model" in st.session_state:
        model = st.session_state["eval_chk_model"]
        d = st.session_state["eval_chk_data"]
        r = st.session_state["eval_chk_results"]
        
        X_train, y_train = d["X_train"], d["y_train"]
        X_test, y_test = d["X_test"], d["y_test"]
        features = d["features"]
        
        scores_tr, scores_te = r["scores_tr"], r["scores_te"]
        y_pred_tr, y_pred_te = r["y_pred_tr"], r["y_pred_te"]
        
        metrics_order = []
        if task_type == "classification":
             metrics_order = ["roc_auc", "pr_auc", "brier", "log_loss", "gini", "ks", 
                              "f1", "precision", "recall", "accuracy", "bal_acc", "mcc"]
        else:
             metrics_order = ["rmse", "mae", "median_ae", "r2", "mse"]    

        # --- Report Layout ---
        st.subheader("Results Comparison: Train vs Test")
        
        tab_met, tab_plot, tab_drift, tab_cluster, tab_bench, tab_stress, tab_exp = st.tabs([
            "Metrics Comparison", 
            "Diagnostic Plots Comparison",
            "Drift Analysis (PSI/KS)",
            "Input Cluster Coverage Check",
            "Benchmarking",
            "Stress Testing",
            "Explainability (SHAP)"
        ])


        
        # 1. Metrics Comparison
        with tab_met:
             c_m1, c_m2 = st.columns(2)
             
             def _render_metrics(col, title, s_dict):
                 col.write(f"**{title}**")
                 res = []
                 for m_name in metrics_order:
                     if m_name in s_dict:
                         res.append({"Metric": m_name, "Score": s_dict[m_name]})
                 # Fallback
                 for m_name, score in s_dict.items():
                     if m_name not in metrics_order:
                         res.append({"Metric": m_name, "Score": score})
                 
                 if res:
                     col.dataframe(pd.DataFrame(res).style.format({"Score": "{:.4f}"}))
                 else:
                     col.write("No metrics.")
            
             _render_metrics(c_m1, "TRAIN Metrics", scores_tr)
             _render_metrics(c_m2, "TEST Metrics", scores_te)

        # 2. Plots Comparison
        with tab_plot:
             eval_imgs = {} # Init capture dict
             c_p1, c_p2 = st.columns(2)
             
             # Generic plotter to avoid duplication
             def _plot_diagnostics(col, title, y_true, y_pred, X_in, img_dict, prefix):
                 import matplotlib.pyplot as plt
                 import io
                 col.write(f"**{title}**")
                 if task_type == "classification":
                     from sklearn.metrics import confusion_matrix
                     cm = confusion_matrix(y_true, y_pred)
                     col.write("Confusion Matrix:")
                     col.write(pd.DataFrame(cm, columns=["Pred 0", "Pred 1"], index=["True 0", "True 1"]))
                     col.divider()
                     
                     if hasattr(model, "predict_proba"):
                         try:
                             y_prob = model.predict_proba(X_in)[:, 1]
                             import matplotlib.pyplot as plt
                             from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score
                             
                             with col.expander("Diagnostic Curves", expanded=True):
                                 sub_c1, sub_c2 = col.columns(2) # WARNING: Nested columns might not work well in all themes, but usually OK in expander
                                 
                                 # 1. ROC
                                 fpr, tpr, th_roc = roc_curve(y_true, y_prob)
                                 fig_roc, ax_roc = plt.subplots(figsize=(4, 4))
                                 ax_roc.plot(fpr, tpr, color='blue', lw=2, label=f'AUC={roc_auc_score(y_true, y_prob):.3f}')
                                 ax_roc.plot([0,1], [0,1], 'r--')
                                 ax_roc.set(title="ROC", xlabel="FPR", ylabel="TPR")
                                 ax_roc.legend(loc="lower right")
                                 sub_c1.pyplot(fig_roc)
                                 
                                 # Save ROC
                                 import io
                                 buf_roc = io.BytesIO()
                                 fig_roc.savefig(buf_roc, format='png', bbox_inches='tight')
                                 buf_roc.seek(0)
                                 img_dict[f"{prefix}_roc"] = buf_roc.read()
                                 plt.close(fig_roc)
                                 
                                 # 2. PR
                                 prec, rec, th_pr = precision_recall_curve(y_true, y_prob)
                                 fig_pr, ax_pr = plt.subplots(figsize=(4, 4))
                                 ax_pr.plot(rec, prec, color='green', lw=2)
                                 ax_pr.set(title="PR Curve", xlabel="Recall", ylabel="Precision")
                                 sub_c2.pyplot(fig_pr)
                                 
                                 # Save PR
                                 buf_pr = io.BytesIO()
                                 fig_pr.savefig(buf_pr, format='png', bbox_inches='tight')
                                 buf_pr.seek(0)
                                 img_dict[f"{prefix}_pr"] = buf_pr.read()
                                 plt.close(fig_pr)
                                 
                                 sub_c3, sub_c4 = col.columns(2)
                                 
                                 # F1 vs Threshold
                                 with np.errstate(divide='ignore', invalid='ignore'):
                                     f1 = 2 * (prec * rec) / (prec + rec)
                                 f1 = np.nan_to_num(f1)
                                 if len(th_pr) < len(f1): f1_plot = f1[:len(th_pr)]
                                 else: f1_plot = f1
                                 
                                 fig_f1, ax_f1 = plt.subplots(figsize=(4, 4))
                                 ax_f1.plot(th_pr, f1_plot, color='purple', lw=2)
                                 ax_f1.set(title="F1 vs Threshold", xlabel="Th", ylabel="F1")
                                 sub_c3.pyplot(fig_f1)
                                 
                                 # Save F1
                                 buf_f1 = io.BytesIO()
                                 fig_f1.savefig(buf_f1, format='png', bbox_inches='tight')
                                 buf_f1.seek(0)
                                 img_dict[f"{prefix}_f1"] = buf_f1.read()
                                 plt.close(fig_f1)
                                 
                                 # Classic CDF
                                 # Inline definition to avoid missing ref
                                 def _plot_cdf_ks_local(y_true, y_prob):
                                     y0 = np.sort(y_prob[y_true==0])
                                     y1 = np.sort(y_prob[y_true==1])
                                     n0 = len(y0); n1 = len(y1)
                                     y_axis0 = np.arange(1, n0+1) / n0
                                     y_axis1 = np.arange(1, n1+1) / n1
                                     fig, ax = plt.subplots(figsize=(4, 4))
                                     ax.plot(y0, y_axis0, 'r', label='Neg CDF', lw=2)
                                     ax.plot(y1, y_axis1, 'b', label='Pos CDF', lw=2)
                                     x_base = np.linspace(0, 1, 1000)
                                     c0 = np.interp(x_base, y0, y_axis0, left=0, right=1)
                                     c1 = np.interp(x_base, y1, y_axis1, left=0, right=1)
                                     d = np.abs(c0 - c1)
                                     ks_x = x_base[np.argmax(d)]
                                     ax.plot([ks_x, ks_x], [c1[np.argmax(d)], c0[np.argmax(d)]], 'k--', label=f'KS={np.max(d):.3f}')
                                     ax.legend(); ax.grid(alpha=0.3); ax.set_title("CDF KS Plot")
                                     return fig

                                 fig_cdf = _plot_cdf_ks_local(y_true, y_prob)
                                 sub_c4.pyplot(fig_cdf)
                                 
                                 # Save CDF
                                 buf_cdf = io.BytesIO()
                                 fig_cdf.savefig(buf_cdf, format='png', bbox_inches='tight')
                                 buf_cdf.seek(0)
                                 img_dict[f"{prefix}_ks_cdf"] = buf_cdf.read()
                                 plt.close(fig_cdf)

                         except Exception as e:
                             col.error(f"Error plotting curves: {e}")
                 else:
                     # Regression Plots - Enhanced
                     import scipy.stats as scipy_stats
                     
                     residuals = y_true - y_pred
                     r2_val = 1 - (np.sum(residuals**2) / np.sum((y_true - np.mean(y_true))**2)) if len(y_true) > 1 else 0
                     
                     if len(y_true) > 1000:
                         sample_idx = np.random.RandomState(42).choice(len(y_true), 1000, replace=False)
                         y_true_s, y_pred_s, residuals_s = y_true[sample_idx], y_pred[sample_idx], residuals[sample_idx]
                     else:
                         y_true_s, y_pred_s, residuals_s = y_true, y_pred, residuals
                     
                     with col.expander(f"Regression Diagnostics", expanded=True):
                         # Row 1: Pred vs Actual & Residuals vs Predicted
                         sub_c1, sub_c2 = st.columns(2)
                         
                         with sub_c1:
                             fig_pva, ax_pva = plt.subplots(figsize=(4, 4))
                             ax_pva.scatter(y_true_s, y_pred_s, alpha=0.4, s=15, c='steelblue')
                             lims = [min(y_true_s.min(), y_pred_s.min()), max(y_true_s.max(), y_pred_s.max())]
                             ax_pva.plot(lims, lims, 'r--', lw=2)
                             ax_pva.set_xlabel("Actual", fontsize=9)
                             ax_pva.set_ylabel("Predicted", fontsize=9)
                             ax_pva.set_title(f"Pred vs Actual (R²={r2_val:.3f})", fontsize=10, fontweight='bold')
                             ax_pva.grid(alpha=0.3)
                             st.pyplot(fig_pva)
                             # Save
                             buf = io.BytesIO()
                             fig_pva.savefig(buf, format='png', bbox_inches='tight')
                             buf.seek(0)
                             img_dict[f"{prefix}_pred_actual"] = buf.read()
                             plt.close(fig_pva)
                         
                         with sub_c2:
                             fig_rvp, ax_rvp = plt.subplots(figsize=(4, 4))
                             ax_rvp.scatter(y_pred_s, residuals_s, alpha=0.4, s=15, c='darkorange')
                             ax_rvp.axhline(y=0, color='red', linestyle='--', lw=2)
                             ax_rvp.set_xlabel("Predicted", fontsize=9)
                             ax_rvp.set_ylabel("Residual", fontsize=9)
                             ax_rvp.set_title("Residuals vs Predicted", fontsize=10, fontweight='bold')
                             ax_rvp.grid(alpha=0.3)
                             st.pyplot(fig_rvp)
                             # Save
                             buf = io.BytesIO()
                             fig_rvp.savefig(buf, format='png', bbox_inches='tight')
                             buf.seek(0)
                             img_dict[f"{prefix}_residuals"] = buf.read()
                             plt.close(fig_rvp)
                         
                         # Row 2: Residual Histogram & Q-Q Plot
                         sub_c3, sub_c4 = st.columns(2)
                         
                         with sub_c3:
                             fig_hist, ax_hist = plt.subplots(figsize=(4, 3.5))
                             ax_hist.hist(residuals, bins=25, edgecolor='black', alpha=0.7, color='teal')
                             ax_hist.axvline(x=0, color='red', linestyle='--', lw=2)
                             ax_hist.set_xlabel("Residual", fontsize=9)
                             ax_hist.set_ylabel("Frequency", fontsize=9)
                             ax_hist.set_title("Residual Distribution", fontsize=10, fontweight='bold')
                             ax_hist.grid(alpha=0.3)
                             st.pyplot(fig_hist)
                             # Save
                             buf = io.BytesIO()
                             fig_hist.savefig(buf, format='png', bbox_inches='tight')
                             buf.seek(0)
                             img_dict[f"{prefix}_residual_hist"] = buf.read()
                             plt.close(fig_hist)
                         
                         with sub_c4:
                             fig_qq, ax_qq = plt.subplots(figsize=(4, 3.5))
                             scipy_stats.probplot(residuals, dist="norm", plot=ax_qq)
                             ax_qq.set_title("Q-Q Plot (Normality)", fontsize=10, fontweight='bold')
                             ax_qq.grid(alpha=0.3)
                             st.pyplot(fig_qq)
                             # Save
                             buf = io.BytesIO()
                             fig_qq.savefig(buf, format='png', bbox_inches='tight')
                             buf.seek(0)
                             img_dict[f"{prefix}_qq"] = buf.read()
                             plt.close(fig_qq)

             # Render Side by Side
             _plot_diagnostics(c_p1, "TRAIN Diagnostics", y_train, y_pred_tr, X_train, eval_imgs, "train")
             _plot_diagnostics(c_p2, "TEST Diagnostics", y_test, y_pred_te, X_test, eval_imgs, "test")
             
             # Save Evaluation Results + Images to Report Buffer
             eval_payload = {
                 "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                 "task_type": task_type,
                 "metrics_train": scores_tr,
                 "metrics_test": scores_te,
                 "images": eval_imgs
             }
             _update_report_buffer("evaluation", eval_payload)
             # Note: Drift and Stress save individually, but this covers the main comparison plots.
             
        # 3. Drift Analysis (PSI + KS) - Using tanml.analysis module
        with tab_drift:
            st.markdown("### Feature Drift Analysis")
            st.caption("Measures the shift in feature distributions between **Training** (Expected) and **Testing** (Actual) datasets. Uses both **PSI** and **KS Statistic** for robust drift detection.")
            
            # Use the analysis module for drift calculations
            from tanml.analysis.drift import calculate_psi, calculate_ks
            
            drift_results = []
            # Calculate PSI and KS for all numeric features
            for col in features:
                if pd.api.types.is_numeric_dtype(X_train[col]):
                    train_vals = X_train[col].dropna()
                    test_vals = X_test[col].dropna()
                    
                    psi = calculate_psi(train_vals, test_vals)
                    ks_stat, ks_pval = calculate_ks(train_vals, test_vals)
                    
                    # PSI Status
                    psi_status = "🟢 Stable"
                    if psi > 0.2: psi_status = "🔴 Critical"
                    elif psi > 0.1: psi_status = "🟠 Moderate"
                    
                    # KS Status
                    ks_status = "🟢 Stable"
                    if ks_stat > 0.3: ks_status = "🔴 Critical"
                    elif ks_stat > 0.2: ks_status = "🟠 Moderate"
                    elif ks_stat > 0.1: ks_status = "🟡 Minor"
                    
                    drift_results.append({
                        "Feature": col,
                        "PSI": psi,
                        "PSI Status": psi_status,
                        "KS Stat": ks_stat,
                        "KS p-value": ks_pval,
                        "KS Status": ks_status
                    })
            
            if drift_results:
                df_drift = pd.DataFrame(drift_results).sort_values("KS Stat", ascending=False)
                
                # Legend
                st.markdown("""
                **Thresholds:**
                - **PSI**: 🟢 < 0.1 (Stable), 🟠 0.1-0.2 (Moderate), 🔴 > 0.2 (Critical)
                - **KS**: 🟢 < 0.1 (Stable), 🟡 0.1-0.2 (Minor), 🟠 0.2-0.3 (Moderate), 🔴 > 0.3 (Critical)
                """)
                
                # Visual style
                st.dataframe(
                    df_drift.style.format({"PSI": "{:.4f}", "KS Stat": "{:.4f}", "KS p-value": "{:.4f}"})
                )
                
                # Plot top drift feature
                st.write("#### Top Drifting Feature Distribution")
                top_drift = df_drift.iloc[0]["Feature"]
                
                # Generate Matplotlib Figure for Report
                import matplotlib.pyplot as plt
                
                fig_drift, axes = plt.subplots(1, 3, figsize=(14, 4))
                
                # Train Distribution
                axes[0].hist(X_train[top_drift].dropna(), bins=20, alpha=0.7, color='blue', edgecolor='black', label='Train')
                axes[0].set_title(f"Train: {top_drift}")
                axes[0].set_xlabel("Value")
                axes[0].set_ylabel("Frequency")
                
                # Test Distribution
                axes[1].hist(X_test[top_drift].dropna(), bins=20, alpha=0.7, color='orange', edgecolor='black', label='Test')
                axes[1].set_title(f"Test: {top_drift}")
                axes[1].set_xlabel("Value")
                
                # CDF Comparison (KS visualization)
                train_sorted = np.sort(X_train[top_drift].dropna())
                test_sorted = np.sort(X_test[top_drift].dropna())
                train_cdf = np.arange(1, len(train_sorted)+1) / len(train_sorted)
                test_cdf = np.arange(1, len(test_sorted)+1) / len(test_sorted)
                
                axes[2].plot(train_sorted, train_cdf, 'b-', lw=2, label='Train CDF')
                axes[2].plot(test_sorted, test_cdf, 'orange', lw=2, label='Test CDF')
                axes[2].set_title(f"{top_drift}: CDF (KS={df_drift.iloc[0]['KS Stat']:.3f})")
                axes[2].set_xlabel("Value")
                axes[2].set_ylabel("CDF")
                axes[2].legend()
                axes[2].grid(alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig_drift)
                
                # Save Drift Image
                import io
                buf_drift = io.BytesIO()
                fig_drift.savefig(buf_drift, format='png', bbox_inches='tight')
                buf_drift.seek(0)
                plt.close(fig_drift)
                
                # Save to Buffer
                _update_report_buffer("drift_images", {"top_distribution": buf_drift.read()})
            else:
                st.warning("No numeric features available for Drift Analysis.")
            
            # --- AUTO-SAVE DRIFT ---
            if drift_results:
                 _update_report_buffer("drift", drift_results)
                 st.toast("Drift Analysis saved to Report!", icon="🌊")

        # 4. Input Cluster Coverage Check - Using tanml.analysis module
        with tab_cluster:
            st.markdown("### Input Cluster Coverage Check")
            st.caption("Evaluates how well the **Testing Data** covers the input space defined by the **Training Data** clusters. Low coverage may indicate the model is being applied to out-of-distribution (OOD) samples.")
            
            # User selects number of clusters
            col_clust_opt, col_clust_btn = st.columns([1, 2])
            with col_clust_opt:
                n_clusters_input = st.slider("Number of Clusters", min_value=2, max_value=20, value=5, 
                                            help="Choose the number of K-Means clusters to partition the training data into.")
            
            with col_clust_btn:
                run_cluster = st.button("🎯 Run Cluster Coverage Check", type="secondary", key="btn_cluster_cov")
            
            if run_cluster:
                with st.spinner("Running K-Means Clustering..."):
                    try:
                        # Use the analysis module for clustering
                        from tanml.analysis.clustering import analyze_cluster_coverage
                        
                        cluster_results_raw = analyze_cluster_coverage(
                            X_train=X_train,
                            X_test=X_test,
                            n_clusters=n_clusters_input,
                        )
                        
                        # Build UI-friendly results from analysis module output
                        n_clusters = cluster_results_raw["n_clusters"]
                        coverage_pct = cluster_results_raw["coverage_pct"]
                        uncovered_count = cluster_results_raw["uncovered_count"]
                        cluster_dist = cluster_results_raw["cluster_distribution"]
                        
                        # Calculate OOD metrics (additional UI-specific logic)
                        ood_pct = 100 - coverage_pct
                        ood_indices = cluster_results_raw["uncovered_indices"]
                        
                        # Build cluster summary table
                        cluster_summary = []
                        for c in range(n_clusters):
                            if c in cluster_dist:
                                train_c = cluster_dist[c]["train_count"]
                                test_c = cluster_dist[c]["test_count"]
                            else:
                                train_c, test_c = 0, 0
                            cluster_summary.append({
                                "Cluster": c,
                                "Train Count": int(train_c),
                                "Train %": f"{train_c / len(X_train) * 100:.1f}%",
                                "Test Count": int(test_c),
                                "Test %": f"{test_c / len(X_test) * 100:.1f}%",
                                "Status": "✓ Covered" if test_c > 0 else "✗ Uncovered"
                            })
                        
                        cluster_results = {
                            "n_clusters": n_clusters,
                            "coverage_pct": coverage_pct,
                            "covered_clusters": n_clusters - len([c for c in cluster_dist.values() if c["test_count"] == 0]),
                            "uncovered_clusters": len([c for c in cluster_dist.values() if c["test_count"] == 0]),
                            "ood_pct": ood_pct,
                            "ood_count": uncovered_count,
                            "ood_indices": ood_indices,
                            "cluster_summary": cluster_summary
                        }
                        
                        # Store OOD samples data for download
                        if uncovered_count > 0 and ood_indices:
                            ood_df = X_test.iloc[ood_indices].copy()
                            st.session_state["eval_ood_samples"] = ood_df
                        
                        # Store PCA data for visualization (from analysis module)
                        train_pca = np.array(cluster_results_raw.get("train_pca", []))
                        test_pca = np.array(cluster_results_raw.get("test_pca", []))
                        centers_pca = np.array(cluster_results_raw.get("cluster_centers_pca", []))
                        
                        st.session_state["eval_cluster_pca"] = {
                            "train_pca": train_pca,
                            "test_pca": test_pca,
                            "centers_pca": centers_pca,
                            "train_labels": cluster_results_raw.get("train_labels", []),
                            "test_labels": cluster_results_raw.get("test_labels", []),
                            "n_clusters": n_clusters,
                            "total_train": len(X_train),
                            "total_test": len(X_test)
                        }
                        
                        # Save to session state
                        st.session_state["eval_cluster_results"] = cluster_results
                        
                        st.success(f"Cluster Coverage Check Complete!")
                        
                    except Exception as e:
                        st.error(f"Cluster Coverage Check Failed: {e}")
            
            # Display Results if available
            if "eval_cluster_results" in st.session_state:
                cr = st.session_state["eval_cluster_results"]
                
                # Key Metrics
                st.write("#### Coverage Summary")
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Total Clusters", cr["n_clusters"])
                k2.metric("Coverage", f"{cr['coverage_pct']:.1f}%", 
                         delta=None if cr['coverage_pct'] >= 90 else f"-{100 - cr['coverage_pct']:.1f}%")
                k3.metric("Uncovered Clusters", cr["uncovered_clusters"],
                         delta_color="inverse" if cr["uncovered_clusters"] > 0 else "off")
                k4.metric("Potential OOD Samples", f"{cr['ood_pct']:.1f}%",
                         delta=f"{cr['ood_count']} samples" if cr['ood_count'] > 0 else None,
                         delta_color="inverse" if cr['ood_count'] > 0 else "off")
                
                # Interpretation
                st.write("")
                if cr['coverage_pct'] >= 95:
                    st.success("✅ **Excellent Coverage**: Test data covers nearly all training input space clusters.")
                elif cr['coverage_pct'] >= 80:
                    st.warning("⚠️ **Good Coverage**: Most clusters are covered, but some regions of the input space are not represented in testing.")
                else:
                    st.error("🚨 **Poor Coverage**: Test data does not adequately cover the training input space. Model may be applied to unfamiliar data patterns.")
                
                if cr['ood_pct'] > 10:
                    st.warning(f"⚠️ **OOD Alert**: {cr['ood_pct']:.1f}% of test samples are far from any training cluster center (potential out-of-distribution data).")
                
                # OOD Samples Viewer & Download
                if cr['ood_count'] > 0 and "eval_ood_samples" in st.session_state:
                    with st.expander(f"📋 View & Download OOD Samples ({cr['ood_count']} samples)", expanded=False):
                        ood_df = st.session_state["eval_ood_samples"]
                        
                        st.write("**Out-of-Distribution Samples** (sorted by distance from nearest cluster center)")
                        st.caption("These test samples are far from any training cluster center, suggesting they may represent data patterns not seen during training.")
                        
                        # Show dataframe
                        display_df = ood_df.sort_values('_ood_distance', ascending=False).reset_index(drop=True)
                        st.dataframe(display_df, use_container_width=True, height=300)
                        
                        # Download button
                        csv_data = display_df.to_csv(index=False)
                        st.download_button(
                            label="⬇️ Download OOD Samples (CSV)",
                            data=csv_data,
                            file_name="ood_samples.csv",
                            mime="text/csv",
                            type="primary"
                        )
                
                # Cluster Summary Table
                st.write("#### Cluster Distribution")
                st.dataframe(
                    pd.DataFrame(cr["cluster_summary"]).style
                    .map(lambda v: "color: green" if "Covered" in str(v) else "color: red" if "Uncovered" in str(v) else "", subset=["Status"])
                )
                
                # Visualization
                st.write("#### Train vs Test Cluster Distribution (Share of Records)")
                import matplotlib.pyplot as plt
                
                fig_clust, ax_clust = plt.subplots(figsize=(12, 5))
                x_pos = np.arange(cr["n_clusters"])
                width = 0.35
                
                # Calculate percentages instead of counts
                train_counts = [d["Train Count"] for d in cr["cluster_summary"]]
                test_counts = [d["Test Count"] for d in cr["cluster_summary"]]
                total_train = sum(train_counts)
                total_test = sum(test_counts)
                
                train_pcts = [c / total_train * 100 if total_train > 0 else 0 for c in train_counts]
                test_pcts = [c / total_test * 100 if total_test > 0 else 0 for c in test_counts]
                
                # Create bars
                bars_train = ax_clust.bar(x_pos - width/2, train_pcts, width, label='Train', color='steelblue', alpha=0.8)
                bars_test = ax_clust.bar(x_pos + width/2, test_pcts, width, label='Test', color='darkorange', alpha=0.8)
                
                # Add annotations above each bar
                for bar in bars_train:
                    height = bar.get_height()
                    ax_clust.annotate(f'{height:.1f}%',
                                     xy=(bar.get_x() + bar.get_width() / 2, height),
                                     xytext=(0, 3), textcoords="offset points",
                                     ha='center', va='bottom', fontsize=8, fontweight='bold', color='steelblue')
                
                for bar in bars_test:
                    height = bar.get_height()
                    ax_clust.annotate(f'{height:.1f}%',
                                     xy=(bar.get_x() + bar.get_width() / 2, height),
                                     xytext=(0, 3), textcoords="offset points",
                                     ha='center', va='bottom', fontsize=8, fontweight='bold', color='darkorange')
                
                ax_clust.set_xlabel('Cluster ID', fontsize=10)
                ax_clust.set_ylabel('Share of Records (%)', fontsize=10)
                ax_clust.set_title('Cluster Distribution: Train vs Test (Percentage)', fontsize=12)
                ax_clust.set_xticks(x_pos)
                ax_clust.set_xticklabels([f'C{i}' for i in x_pos])
                ax_clust.legend(loc='upper right')
                ax_clust.grid(axis='y', alpha=0.3)
                ax_clust.set_ylim(0, max(max(train_pcts), max(test_pcts)) * 1.2)  # Add headroom for annotations
                
                st.pyplot(fig_clust)
                
                # Save image for report
                import io
                buf_clust = io.BytesIO()
                fig_clust.savefig(buf_clust, format='png', bbox_inches='tight')
                buf_clust.seek(0)
                plt.close(fig_clust)
                
                # 2nd Diagram: PCA Scatter Plot
                st.write("#### Cluster Space Visualization (2D Projection)")
                st.caption("💡 **How to read**: Gray = Training data. Colored squares = Test data. Each centroid shows **Train% / Test% (ratio)**.")
                
                if "eval_cluster_pca" in st.session_state:
                    pca_data = st.session_state["eval_cluster_pca"]
                    train_pca = pca_data["train_pca"]
                    test_pca = pca_data["test_pca"]
                    centers_pca = pca_data["centers_pca"]
                    test_labels = pca_data["test_labels"]
                    n_clusters = pca_data["n_clusters"]
                    explained_var = pca_data.get("explained_variance", [0, 0])
                    train_counts = pca_data.get("train_cluster_counts", {})
                    test_counts = pca_data.get("test_cluster_counts", {})
                    total_train = pca_data.get("total_train", 1)
                    total_test = pca_data.get("total_test", 1)
                    
                    fig_pca, ax_pca = plt.subplots(figsize=(12, 8))
                    
                    # Color palette for clusters
                    cluster_colors = plt.cm.Set1(np.linspace(0, 1, max(n_clusters, 9)))[:n_clusters]
                    
                    # Plot training points in gray (background, less clutter)
                    ax_pca.scatter(train_pca[:, 0], train_pca[:, 1], 
                                  c='lightgray', alpha=0.25, s=10, label='Train Data', zorder=1)
                    
                    # Plot test points with cluster-specific colors (simpler - no legend per cluster)
                    for c in range(n_clusters):
                        mask = test_labels == c
                        if mask.sum() > 0:
                            ax_pca.scatter(test_pca[mask, 0], test_pca[mask, 1],
                                          c=[cluster_colors[c]], alpha=0.85, s=60, marker='s', 
                                          edgecolors='black', linewidths=0.5, zorder=3)
                    
                    # Plot cluster centers with coverage annotations
                    for i, (cx, cy) in enumerate(centers_pca):
                        # Get counts
                        tr_count = train_counts.get(i, 0)
                        te_count = test_counts.get(i, 0)
                        tr_pct = tr_count / total_train * 100 if total_train > 0 else 0
                        te_pct = te_count / total_test * 100 if total_test > 0 else 0
                        ratio = te_pct / tr_pct if tr_pct > 0 else 0
                        
                        # Plot center marker
                        ax_pca.scatter(cx, cy, c=[cluster_colors[i]], s=400, marker='X', 
                                      edgecolors='black', linewidths=2, zorder=5)
                        
                        # Annotation with coverage stats
                        if te_count > 0:
                            label_text = f"C{i}\nTr:{tr_pct:.0f}% Te:{te_pct:.0f}%\n({ratio:.1f}×)"
                        else:
                            label_text = f"C{i}\nTr:{tr_pct:.0f}% Te:0%\n(No coverage)"
                        
                        ax_pca.annotate(label_text, (cx, cy), fontsize=8, fontweight='bold',
                                       xytext=(12, 12), textcoords='offset points',
                                       bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                                                edgecolor=cluster_colors[i], alpha=0.95),
                                       ha='left', va='bottom')
                    
                    # Axes with explained variance
                    var1 = explained_var[0] * 100 if len(explained_var) > 0 else 0
                    var2 = explained_var[1] * 100 if len(explained_var) > 1 else 0
                    ax_pca.set_xlabel(f'PC1 ({var1:.1f}% variance)', fontsize=11, fontweight='bold')
                    ax_pca.set_ylabel(f'PC2 ({var2:.1f}% variance)', fontsize=11, fontweight='bold')
                    ax_pca.set_title('Test Data Coverage in Cluster Space', fontsize=13, fontweight='bold')
                    
                    # Simplified legend
                    from matplotlib.lines import Line2D
                    legend_elements = [
                        Line2D([0], [0], marker='o', color='white', markerfacecolor='lightgray', 
                               markersize=8, label='Train Data'),
                        Line2D([0], [0], marker='s', color='white', markerfacecolor='steelblue', 
                               markeredgecolor='black', markersize=10, label='Test Data'),
                        Line2D([0], [0], marker='X', color='white', markerfacecolor='red', 
                               markeredgecolor='black', markersize=12, label='Cluster Center')
                    ]
                    ax_pca.legend(handles=legend_elements, loc='upper right', fontsize=9, framealpha=0.95)
                    
                    ax_pca.grid(alpha=0.2, linestyle='-')
                    plt.tight_layout()
                    
                    st.pyplot(fig_pca)
                    
                    # Save PCA image for report
                    buf_pca = io.BytesIO()
                    fig_pca.savefig(buf_pca, format='png', bbox_inches='tight', dpi=150)
                    buf_pca.seek(0)
                    plt.close(fig_pca)
                    
                    # Save to report buffer (both images)
                    _update_report_buffer("cluster_coverage", cr)
                    _update_report_buffer("cluster_images", {
                        "distribution": buf_clust.read(),
                        "pca_scatter": buf_pca.read()
                    })
                else:
                    # Fallback: just save distribution chart
                    _update_report_buffer("cluster_coverage", cr)
                    _update_report_buffer("cluster_images", {"distribution": buf_clust.read()})
                
                st.toast("Cluster Coverage saved to Report!", icon="🎯")

        # 5. Benchmarking
        with tab_bench:
            st.markdown("### Benchmarking: Compare Your Model vs Baseline Models")
            st.caption("Select one or more baseline models to compare against your trained model.")
            
            # Model selection based on task type
            if task_type == "classification":
                available_models = {
                    "Logistic Regression (statsmodels)": "sm_logit",
                    "Logistic Regression (sklearn)": "sk_logistic",
                    "Random Forest": "sk_rf",
                    "Decision Tree": "sk_dt",
                    "Naive Bayes": "sk_nb",
                    "Dummy Classifier (Most Frequent)": "sk_dummy"
                }
            else:  # regression
                available_models = {
                    "OLS Regression (statsmodels)": "sm_ols",
                    "Linear Regression (sklearn)": "sk_linear",
                    "Ridge Regression": "sk_ridge",
                    "Random Forest": "sk_rf",
                    "Decision Tree": "sk_dt",
                    "Dummy Regressor (Mean)": "sk_dummy"
                }
            
            selected_models = st.multiselect(
                "Select Baseline Models to Compare",
                options=list(available_models.keys()),
                default=[list(available_models.keys())[0]],
                help="Choose one or more models to benchmark against"
            )
            
            if st.button("🔬 Run Benchmark Comparison", type="secondary", key="btn_benchmark"):
                if not selected_models:
                    st.warning("Please select at least one baseline model.")
                else:
                    with st.spinner("Training baseline models..."):
                        try:
                            import statsmodels.api as sm
                            from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
                            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
                            from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
                            from sklearn.naive_bayes import GaussianNB
                            from sklearn.dummy import DummyClassifier, DummyRegressor
                            from sklearn.metrics import (
                                roc_auc_score, f1_score, accuracy_score, precision_score, recall_score,
                                mean_squared_error, mean_absolute_error, r2_score
                            )
                            
                            # Prepare data
                            X_tr_num = X_train.select_dtypes(include=np.number).fillna(0)
                            X_te_num = X_test.select_dtypes(include=np.number).fillna(0)
                            
                            benchmark_results = {"your_model": {}, "baselines": {}}
                            
                            # Your model's metrics
                            if task_type == "classification":
                                benchmark_results["your_model"] = {
                                    "roc_auc": scores_te.get("roc_auc", 0),
                                    "f1": scores_te.get("f1", 0),
                                    "accuracy": scores_te.get("accuracy", 0),
                                    "precision": scores_te.get("precision", 0),
                                    "recall": scores_te.get("recall", 0)
                                }
                                higher_better = ["roc_auc", "f1", "accuracy", "precision", "recall"]
                            else:
                                benchmark_results["your_model"] = {
                                    "rmse": scores_te.get("rmse", 0),
                                    "mae": scores_te.get("mae", 0),
                                    "r2": scores_te.get("r2", 0)
                                }
                                higher_better = ["r2"]
                            
                            # Train each selected baseline
                            for model_name in selected_models:
                                model_key = available_models[model_name]
                                try:
                                    if task_type == "classification":
                                        # Initialize model
                                        if model_key == "sm_logit":
                                            X_sm = sm.add_constant(X_tr_num)
                                            X_sm_te = sm.add_constant(X_te_num)
                                            mdl = sm.Logit(y_train, X_sm).fit(disp=0, maxiter=100)
                                            pred_prob = mdl.predict(X_sm_te)
                                            pred = (pred_prob >= 0.5).astype(int)
                                        elif model_key == "sk_logistic":
                                            mdl = LogisticRegression(max_iter=200, random_state=42)
                                            mdl.fit(X_tr_num, y_train)
                                            pred = mdl.predict(X_te_num)
                                            pred_prob = mdl.predict_proba(X_te_num)[:, 1]
                                        elif model_key == "sk_rf":
                                            mdl = RandomForestClassifier(n_estimators=50, random_state=42)
                                            mdl.fit(X_tr_num, y_train)
                                            pred = mdl.predict(X_te_num)
                                            pred_prob = mdl.predict_proba(X_te_num)[:, 1]
                                        elif model_key == "sk_dt":
                                            mdl = DecisionTreeClassifier(random_state=42)
                                            mdl.fit(X_tr_num, y_train)
                                            pred = mdl.predict(X_te_num)
                                            pred_prob = mdl.predict_proba(X_te_num)[:, 1]
                                        elif model_key == "sk_nb":
                                            mdl = GaussianNB()
                                            mdl.fit(X_tr_num, y_train)
                                            pred = mdl.predict(X_te_num)
                                            pred_prob = mdl.predict_proba(X_te_num)[:, 1]
                                        elif model_key == "sk_dummy":
                                            mdl = DummyClassifier(strategy="most_frequent")
                                            mdl.fit(X_tr_num, y_train)
                                            pred = mdl.predict(X_te_num)
                                            pred_prob = np.full(len(y_test), y_train.mean())
                                        
                                        # Calculate metrics
                                        benchmark_results["baselines"][model_name] = {
                                            "roc_auc": roc_auc_score(y_test, pred_prob) if len(np.unique(pred_prob)) > 1 else 0.5,
                                            "f1": f1_score(y_test, pred),
                                            "accuracy": accuracy_score(y_test, pred),
                                            "precision": precision_score(y_test, pred, zero_division=0),
                                            "recall": recall_score(y_test, pred, zero_division=0)
                                        }
                                    else:  # Regression
                                        if model_key == "sm_ols":
                                            X_sm = sm.add_constant(X_tr_num)
                                            X_sm_te = sm.add_constant(X_te_num)
                                            mdl = sm.OLS(y_train, X_sm).fit()
                                            pred = mdl.predict(X_sm_te)
                                        elif model_key == "sk_linear":
                                            mdl = LinearRegression()
                                            mdl.fit(X_tr_num, y_train)
                                            pred = mdl.predict(X_te_num)
                                        elif model_key == "sk_ridge":
                                            mdl = Ridge()
                                            mdl.fit(X_tr_num, y_train)
                                            pred = mdl.predict(X_te_num)
                                        elif model_key == "sk_rf":
                                            mdl = RandomForestRegressor(n_estimators=50, random_state=42)
                                            mdl.fit(X_tr_num, y_train)
                                            pred = mdl.predict(X_te_num)
                                        elif model_key == "sk_dt":
                                            mdl = DecisionTreeRegressor(random_state=42)
                                            mdl.fit(X_tr_num, y_train)
                                            pred = mdl.predict(X_te_num)
                                        elif model_key == "sk_dummy":
                                            mdl = DummyRegressor(strategy="mean")
                                            mdl.fit(X_tr_num, y_train)
                                            pred = mdl.predict(X_te_num)
                                        
                                        benchmark_results["baselines"][model_name] = {
                                            "rmse": np.sqrt(mean_squared_error(y_test, pred)),
                                            "mae": mean_absolute_error(y_test, pred),
                                            "r2": r2_score(y_test, pred)
                                        }
                                except Exception as model_err:
                                    st.warning(f"Failed to train {model_name}: {model_err}")
                            
                            benchmark_results["higher_better"] = higher_better
                            benchmark_results["task_type"] = task_type
                            st.session_state["benchmark_results"] = benchmark_results
                            st.success(f"✅ Benchmark complete! Compared against {len(benchmark_results['baselines'])} baseline models.")
                            
                        except Exception as e:
                            st.error(f"Benchmark failed: {e}")
            
            # Display Results
            if "benchmark_results" in st.session_state:
                br = st.session_state["benchmark_results"]
                
                if br.get("baselines"):
                    st.write("#### Test Set Comparison")
                    
                    # Build comparison table
                    metrics = list(br["your_model"].keys())
                    table_data = []
                    
                    for metric in metrics:
                        row = {"Metric": metric.upper(), "Your Model": br["your_model"][metric]}
                        for baseline_name, baseline_metrics in br["baselines"].items():
                            row[baseline_name] = baseline_metrics.get(metric, 0)
                        table_data.append(row)
                    
                    df_compare = pd.DataFrame(table_data)
                    
                    # Determine best model for each metric
                    def highlight_best(row):
                        metric = row["Metric"].lower()
                        values = {k: v for k, v in row.items() if k != "Metric"}
                        
                        if metric in [m.upper() for m in br.get("higher_better", [])]:
                            best_model = max(values, key=values.get)
                        else:
                            best_model = min(values, key=values.get)
                        
                        styles = [""]  # Metric column
                        for col in df_compare.columns[1:]:
                            if col == best_model:
                                styles.append("background-color: #d4edda; font-weight: bold")
                            else:
                                styles.append("")
                        return styles
                    
                    # Format numbers
                    format_dict = {col: "{:.4f}" for col in df_compare.columns if col != "Metric"}
                    styled_df = df_compare.style.apply(highlight_best, axis=1).format(format_dict)
                    st.dataframe(styled_df, use_container_width=True, hide_index=True)
                    
                    # Bar Chart Comparison
                    st.write("#### Performance Comparison Chart")
                    import matplotlib.pyplot as plt
                    
                    fig_bench, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 5))
                    if len(metrics) == 1:
                        axes = [axes]
                    
                    all_models = ["Your Model"] + list(br["baselines"].keys())
                    colors = plt.cm.Set2(np.linspace(0, 1, len(all_models)))
                    
                    # Metrics that should be scaled 0-1
                    proportion_metrics = ["roc_auc", "f1", "accuracy", "precision", "recall", "r2"]
                    
                    for idx, metric in enumerate(metrics):
                        ax = axes[idx]
                        values = [br["your_model"][metric]]
                        values += [br["baselines"][m].get(metric, 0) for m in br["baselines"].keys()]
                        
                        bars = ax.bar(range(len(all_models)), values, color=colors)
                        ax.set_xticks(range(len(all_models)))
                        ax.set_xticklabels([m[:15] + "..." if len(m) > 15 else m for m in all_models], 
                                          rotation=45, ha='right', fontsize=8)
                        ax.set_title(metric.upper(), fontsize=11, fontweight='bold')
                        ax.set_ylabel(metric.upper())
                        
                        # Set appropriate Y-axis limits
                        if metric in proportion_metrics:
                            max_val = max(values) if max(values) > 0 else 1
                            ax.set_ylim(0, max(1.0, max_val * 1.1))
                        else:
                            # For error metrics like RMSE/MAE, scale based on actual values
                            if max(values) > 0:
                                ax.set_ylim(0, max(values) * 1.2)
                        
                        # Highlight best
                        if metric in br.get("higher_better", []):
                            best_idx = values.index(max(values))
                        else:
                            best_idx = values.index(min(values))
                        bars[best_idx].set_edgecolor('gold')
                        bars[best_idx].set_linewidth(3)
                        
                        # Add value labels
                        for bar_idx, bar in enumerate(bars):
                            ax.annotate(f'{values[bar_idx]:.3f}', 
                                       xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                                       xytext=(0, 3), textcoords='offset points',
                                       ha='center', va='bottom', fontsize=7)
                    
                    plt.tight_layout()
                    st.pyplot(fig_bench)
                    
                    # Save chart for report
                    import io
                    buf_bench = io.BytesIO()
                    fig_bench.savefig(buf_bench, format='png', bbox_inches='tight', dpi=150)
                    buf_bench.seek(0)
                    plt.close(fig_bench)
                    
                    # Summary
                    st.write("#### Summary")
                    your_wins = 0
                    for metric in metrics:
                        all_vals = [br["your_model"][metric]] + [br["baselines"][m].get(metric, 0) for m in br["baselines"]]
                        if metric in br.get("higher_better", []):
                            if br["your_model"][metric] == max(all_vals):
                                your_wins += 1
                        else:
                            if br["your_model"][metric] == min(all_vals):
                                your_wins += 1
                    
                    if your_wins == len(metrics):
                        st.success(f"🏆 **Your Model wins** on all {len(metrics)} metrics!")
                    elif your_wins > len(metrics) // 2:
                        st.success(f"🥇 **Your Model leads** on {your_wins}/{len(metrics)} metrics!")
                    elif your_wins > 0:
                        st.info(f"📊 **Mixed Results**: Your model wins on {your_wins}/{len(metrics)} metrics.")
                    else:
                        st.warning(f"⚠️ Baseline models outperform on all metrics. Consider improving your model.")
                    
                    # Save to report buffer
                    _update_report_buffer("benchmark", {
                        "your_model": br["your_model"],
                        "baselines": br["baselines"],
                        "task_type": br["task_type"]
                    })
                    _update_report_buffer("benchmark_images", {"comparison": buf_bench.read()})
                    st.toast("Benchmark saved to Report!", icon="📊")

        # 6. Stress Testing
        with tab_stress:
            st.markdown("### Stress Testing (Robustness)")
            st.caption("Evaluates how model performance degrades when **Testing Data** is perturbed with noise (simulating poor data quality).")
            
            if st.button("Run Stress Test", type="secondary"):
                with st.spinner("Running Stress Checks..."):
                    try:
                        from tanml.checks.stress_test import StressTestCheck
                        
                        # Initialize and Run check on TEST set
                        stress_check = StressTestCheck(model, X_test, y_test, epsilon=0.01, perturb_fraction=0.2)
                        df_stress = stress_check.run()
                        
                        st.write("**Stress Test Results (Perturbation +/- 1%)**")
                        
                        # Hightlight drops
                        if "delta_accuracy" in df_stress.columns:
                            st.dataframe(
                                df_stress.style.format({
                                    "accuracy": "{:.4f}", "auc": "{:.4f}", 
                                    "delta_accuracy": "{:.4f}", "delta_auc": "{:.4f}"
                                }).bar(subset=["delta_accuracy"], color=['#ffcccc', '#ccffcc'], align='zero')
                            )
                        else:
                            st.dataframe(
                                df_stress.style.format({
                                    "rmse": "{:.4f}", "r2": "{:.4f}",
                                    "delta_rmse": "{:.4f}", "delta_r2": "{:.4f}"
                                })
                            )
                            
                    except Exception as e:
                         st.error(f"Stress Test Failed: {e}")
                
            # --- AUTO-SAVE STRESS ---
            if "df_stress" in locals():
                _update_report_buffer("stress", df_stress.to_dict(orient="records"))
                st.toast("Stress Test saved to Report!", icon="💥")

        # 7. Explainability
        with tab_exp:
            st.markdown("### Model Explainability (SHAP)")
            st.caption("Understand feature importance and how features drive the model's predictions.")
            
            if st.button("Run SHAP Analysis", type="secondary"):
                with st.spinner("Running SHAP Check (this may take a minute)..."):
                    try:
                        from tanml.checks.explainability.shap_check import SHAPCheck
                        
                        # We use 100 bg samples and 100 test samples by default for speed
                        scog = {"explainability": {"shap": {"background_sample_size": 50, "test_sample_size": 100}}}
                        
                        shap_check = SHAPCheck(model, X_train, X_test, y_train, y_test, rule_config=scog)
                        res = shap_check.run()
                        
                        if res.get("status") == "ok":
                            plots = res.get("plots", {})
                            
                            c_s1, c_s2 = st.columns(2)
                            
                            with c_s1:
                                if "beeswarm" in plots:
                                    st.image(plots["beeswarm"], caption="SHAP Beeswarm Plot (Global Info)")
                            
                            with c_s2:
                                if "bar" in plots:
                                    st.image(plots["bar"], caption="SHAP Bar Plot (Feature Importance)")
                                    
                            # Top Features Table
                            if "top_features" in res:
                                st.write("**Top Feature Impacts**")
                                st.dataframe(pd.DataFrame(res["top_features"]))
                        else:
                            st.error(f"SHAP Analysis Error: {res.get('status')}")
                            
                    except Exception as e:
                         st.error(f"SHAP Failed: {e}")
            
            # --- AUTO-SAVE EXPL ---
            if "res" in locals() and res.get("status") == "ok":
                 _update_report_buffer("explainability", res)
                 st.toast("SHAP Results saved to Report!", icon="🧠")
        
        # --- DOWNLOAD REPORT BUTTON (EVAL) ---
        st.divider()
        st.subheader("Report")
        if st.button("Generate Evaluation Report 📄"):
            try:
                 buf = st.session_state.get("report_buffer", {})
                 docx_bytes = _generate_eval_report_docx(buf)
                 st.download_button(
                     label="⬇️ Download DOCX",
                     data=docx_bytes,
                     file_name="model_evaluation_report.docx",
                     mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                 )
                 st.success("Report Ready!")
            except Exception as e:
                st.error(f"Report Generation Failed: {e}")




# =========================
# Feature Engineering Hub






