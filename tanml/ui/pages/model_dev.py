# tanml/ui/pages/model_dev.py
"""
Model Development page logic.
"""
from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import scipy.stats as scipy_stats
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, 
                             average_precision_score, brier_score_loss, log_loss, balanced_accuracy_score, matthews_corrcoef,
                             mean_squared_error, mean_absolute_error, median_absolute_error, r2_score,
                             confusion_matrix, roc_curve, precision_recall_curve, auc)
from scipy.stats import ks_2samp

# TanML Internal
from tanml.ui.services.data import _save_upload
from tanml.ui.services.session import _update_report_buffer
from tanml.utils.data_loader import load_dataframe
from tanml.ui.components.forms import render_model_form
from tanml.ui.reports import _generate_dev_report_docx
from tanml.models.registry import build_estimator, infer_task_from_target
from tanml.ui.services.cv import _run_repeated_cv

def render_model_development_page(run_dir):
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    ">
        <h2 style="color: white; margin: 0;">🤖 Model Development</h2>
    </div>
    """, unsafe_allow_html=True)
    st.write("Upload a dedicated Development Dataset to experiment with models.")
    
    # 1. Dedicated Upload
    upl = st.file_uploader("Upload Model Development Dataset", type=["csv", "xlsx", "xls", "parquet", "dta", "sav", "sas7bdat"], key="upl_dev")
    df_dev = None
    if upl:
        path = _save_upload(upl, run_dir)
        if path:
            st.session_state["path_dev"] = str(path)
            try:
                df_dev = load_dataframe(path)
                st.session_state["df_dev"] = df_dev
                st.success(f"Loaded {len(df_dev)} rows.")
            except Exception as e:
                st.error(f"Error loading file: {e}")
    else:
        # Check if already loaded
        df_dev = st.session_state.get("df_dev")
        
    if df_dev is None:
        st.info("Please upload a dataset to proceed.")
        return

    st.divider()
    
    st.subheader("Data Selection")
    all_cols = list(df_dev.columns)
    
    # Target
    curr_target = st.session_state.get("dev_target", all_cols[-1] if all_cols else None)
    # Ensure index valid
    idx = 0
    if curr_target in all_cols:
            idx = all_cols.index(curr_target)
    
    target = st.selectbox("Target Column", all_cols, index=idx, key="dev_target")
    
    # Task Type
    temp_task = infer_task_from_target(df_dev[target])
    st.info(f"Detected Task: **{temp_task.title()}**")
    
    # Features
    possible_feats = [c for c in all_cols if c != target]
    curr_feats = st.session_state.get("dev_features", possible_feats)
    # Intersection to be safe
    default_feats = [f for f in curr_feats if f in possible_feats]
    
    features = st.multiselect("Features", possible_feats, default=default_feats, key="dev_features")
    if not features:
        st.warning("Select features to train on.")
        return

    st.divider()

    st.subheader("Model Config")
    # Reuse helper
    y_sample = df_dev[target]
    # render_model_form returns: library, algo, hp, task_type
    # We pass target_name for cache/key salting
    library, algo, hp, task_type = render_model_form(y_sample, 42, target_name=f"dev_{target}")

    st.divider()
    
    # CV Configuration
    st.subheader("Cross-Validation Config")
    with st.expander("⚙️ CV Settings", expanded=True):
        cv_col1, cv_col2 = st.columns(2)
        with cv_col1:
            n_folds = st.number_input(
                "Number of Folds (K)", 
                min_value=2, max_value=20, value=5, step=1,
                help="Number of splits for K-Fold CV. 5 or 10 are common choices."
            )
        with cv_col2:
            n_repeats = st.number_input(
                "Number of Repeats", 
                min_value=1, max_value=10, value=1, step=1,
                help="Number of times to repeat K-Fold with different random splits. More repeats = more robust estimates but slower."
            )
        
        # Show CV method based on task type
        if task_type == "classification":
            st.info(f"📊 **Method**: Repeated Stratified K-Fold ({n_folds} folds × {n_repeats} repeats = {n_folds * n_repeats} total fits)\n\n*Stratified* ensures each fold maintains the same class distribution as the full dataset.")
        else:
            st.info(f"📊 **Method**: Repeated K-Fold ({n_folds} folds × {n_repeats} repeats = {n_folds * n_repeats} total fits)\n\n*Standard K-Fold* for regression - randomly splits data into K equal parts.")
    
    # 3. Execution (Compute & Store)
    if st.button("Run Development Experiments", type="primary"):
        try:
            # Build estimator
            model = build_estimator(library, algo, hp)
            X = df_dev[features]
            y = df_dev[target]
            
            with st.status("Running Experiments...", expanded=True) as status:
                st.write(f"Running {n_folds}-Fold Cross-Validation (×{n_repeats} repeats)...")
                stats = _run_repeated_cv(model, X, y, task_type, n_splits=n_folds, n_repeats=n_repeats)
                
                st.write("Training Final Model...")
                model.fit(X, y)
                y_pred = model.predict(X)
                y_prob = None
                if hasattr(model, "predict_proba"):
                    try: y_prob = model.predict_proba(X)[:, 1]
                    except: pass
                
                status.update(label="Experiments Complete!", state="complete", expanded=False)
            
            # Save Results to Session State (Persist)
            st.session_state["dev_results"] = {
                "stats": stats,
                "model": model,
                "X": X, "y": y,
                "task_type": task_type,
                "target": target,
                "features": features,
                "y_pred": y_pred,
                "y_prob": y_prob,
                "config": {"library": library, "algorithm": algo, "hp": hp},
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        except Exception as e:
            st.error(f"Experiment Failure: {e}")
            
    # 4. Rendering (Persistent)
    if "dev_results" in st.session_state:
        res = st.session_state["dev_results"]
        # Unpack
        stats = res["stats"]
        model = res["model"]
        X = res["X"]
        y = res["y"]
        task_type = res["task_type"]
        y_pred = res["y_pred"]
        y_prob = res["y_prob"]
        target = res["target"]
        features = res["features"]
        
        st.divider()
        st.subheader("1. Cross-Validation Results")
        
        tab_met, tab_plot = st.tabs(["CV Metrics", "CV Plots"])
        
        # Prepare Data for Report (metrics part)
        cv_metrics_dict = {}
        for m, v in stats.items():
             if isinstance(v, dict) and "mean" in v:
                 cv_metrics_dict[m] = v["mean"]
        
        with tab_met:
            res_data = []
            for m, v in stats.items():
                if m in ["oof", "curves", "threshold_info", "y_probs", "y_trues"]: continue
                if not isinstance(v, dict): continue  # Skip non-dict items
                res_data.append({"Metric": m, "Mean": v.get("mean"), "Std": v.get("std")})
            st.dataframe(pd.DataFrame(res_data).style.format({"Mean": "{:.4f}", "Std": "{:.4f}"}))
            
        with tab_plot:
            cv_imgs = {}
            
            # Inline Helper to avoid scope issues
            def _plot_spaghetti(curve_list, title, xlabel, ylabel, mode="roc"):
                fig, ax = plt.subplots(figsize=(6, 6))
                tprs = []
                aucs = []  # Track AUC for each fold
                base_x = np.linspace(0, 1, 101)
                for item in curve_list:
                    if len(item) == 3: x_val, y_val, _ = item
                    else: x_val, y_val = item
                    if mode == "roc":
                        interp_y = np.interp(base_x, x_val, y_val)
                        interp_y[0] = 0.0
                        # Compute AUC for this fold
                        fold_auc = auc(x_val, y_val)
                        aucs.append(fold_auc)
                    elif mode == "pr":
                        idx = np.argsort(x_val)
                        interp_y = np.interp(base_x, x_val[idx], y_val[idx])
                        # Compute AUC for PR
                        fold_auc = auc(x_val[idx], y_val[idx])
                        aucs.append(fold_auc)
                    elif mode == "thresh":
                         idx = np.argsort(x_val)
                         interp_y = np.interp(base_x, x_val[idx], y_val[idx])
                    tprs.append(interp_y)
                    ax.plot(x_val, y_val, lw=1, alpha=0.3, color='gray')
                
                mean_y = np.mean(tprs, axis=0)
                std_y = np.std(tprs, axis=0)
                if mode == "roc": mean_y[-1] = 1.0
                
                # Create label with AUC if applicable
                if aucs:
                    mean_auc = np.mean(aucs)
                    std_auc = np.std(aucs)
                    label = f'Mean (AUC = {mean_auc:.3f} ± {std_auc:.3f})'
                elif mode == "thresh":
                    # For F1 curve, find max
                    max_f1_idx = np.argmax(mean_y)
                    max_f1 = mean_y[max_f1_idx]
                    best_thresh = base_x[max_f1_idx]
                    label = f'Mean (Max F1 = {max_f1:.3f} @ {best_thresh:.2f})'
                    # Add marker for max F1
                    ax.scatter([best_thresh], [max_f1], color='red', s=100, zorder=5, marker='*')
                    ax.axvline(x=best_thresh, color='red', linestyle='--', alpha=0.5, lw=1)
                else:
                    label = 'Mean'
                
                ax.plot(base_x, mean_y, color='b', lw=2, alpha=0.8, label=label)
                ax.fill_between(base_x, np.maximum(mean_y - std_y, 0), np.minimum(mean_y + std_y, 1), color='grey', alpha=0.2, label='± 1 Std')
                ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
                ax.legend(loc='lower right' if mode == 'roc' else 'lower left')
                ax.grid(alpha=0.3)
                return fig
            
            if "oof" in stats:
                oof = stats["oof"]
                if task_type == "classification":
                     # ROC/PR/F1/KS Logic
                     c_d1, c_d2 = st.columns(2)
                     cv_imgs = {} # Ensure this is cleared
                     
                     # --- CV ROW 1: ROC & PR ---
                     with c_d1:
                         if "roc" in stats["curves"]:
                             data_roc = [(item[0], item[1]) for item in stats["curves"]["roc"]]
                             fig = _plot_spaghetti(data_roc, "ROC Curve", "FPR", "TPR", mode="roc")
                             fig.axes[0].plot([0,1], [0,1], 'r--')
                             st.pyplot(fig)
                             # Save
                             buf = io.BytesIO()
                             fig.savefig(buf, format='png', bbox_inches='tight')
                             buf.seek(0)
                             cv_imgs["roc"] = buf.read()
                             plt.close(fig)
                             
                     with c_d2:
                         if "pr" in stats["curves"]:
                             data_pr = [(item[0], item[1]) for item in stats["curves"]["pr"]]
                             fig = _plot_spaghetti(data_pr, "PR Curve", "Recall", "Precision", mode="pr")
                             st.pyplot(fig)
                             # Save
                             buf = io.BytesIO()
                             fig.savefig(buf, format='png', bbox_inches='tight')
                             buf.seek(0)
                             cv_imgs["pr"] = buf.read()
                             plt.close(fig)

                     # --- CV ROW 2: F1 & KS ---
                     c_d3, c_d4 = st.columns(2)
                     
                     with c_d3:
                         # F1 Curve
                         if "pr" in stats["curves"]:
                             data_f1 = []
                             for (rec, prec, th) in stats["curves"]["pr"]:
                                 with np.errstate(divide='ignore', invalid='ignore'):
                                     f1 = 2 * (prec * rec) / (prec + rec)
                                 f1 = np.nan_to_num(f1)
                                 # Trim f1 to match th length if needed (sklearn differences)
                                 if len(th) < len(f1): f1_ = f1[:len(th)]
                                 else: f1_ = f1
                                 data_f1.append((th, f1_))
                             
                             fig_f1 = _plot_spaghetti(data_f1, "F1 Score vs Threshold", "Threshold", "F1 Score", mode="thresh")
                             st.pyplot(fig_f1)
                             # Save
                             buf = io.BytesIO()
                             fig_f1.savefig(buf, format='png', bbox_inches='tight')
                             buf.seek(0)
                             cv_imgs["f1"] = buf.read()
                             plt.close(fig_f1)
                             
                     with c_d4:
                         # CDF KS Plot
                         if "y_probs" in stats and "y_trues" in stats:
                             fig_ks, ax_ks = plt.subplots(figsize=(6, 6))
                             
                             # Plot CDF for each fold (faint)
                             for y_true_f, y_prob_f in zip(stats["y_trues"], stats["y_probs"]):
                                 try:
                                     y_true_arr = np.array(y_true_f)
                                     y_prob_arr = np.array(y_prob_f)
                                     mask0 = y_true_arr == 0
                                     mask1 = y_true_arr == 1
                                     if mask0.sum() > 0:
                                         y0 = np.sort(y_prob_arr[mask0])
                                         ax_ks.plot(y0, np.arange(1, len(y0)+1)/len(y0), 'r', alpha=0.1, lw=1)
                                     if mask1.sum() > 0:
                                         y1 = np.sort(y_prob_arr[mask1])
                                         ax_ks.plot(y1, np.arange(1, len(y1)+1)/len(y1), 'b', alpha=0.1, lw=1)
                                 except Exception:
                                     pass  # Skip problematic folds
                             
                             # Aggregate all folds for mean CDF
                             try:
                                 # Convert each fold to array before concatenating
                                 y_true_arrays = [np.array(yt) for yt in stats["y_trues"]]
                                 y_prob_arrays = [np.array(yp) for yp in stats["y_probs"]]
                                 all_y_true = np.concatenate(y_true_arrays)
                                 all_y_prob = np.concatenate(y_prob_arrays)
                             except Exception:
                                 # Fallback: use oof data if available
                                 if "oof" in stats:
                                     all_y_true = np.array(stats["oof"]["y_true"])
                                     all_y_prob = np.array(stats["oof"]["y_prob"])
                                 else:
                                     all_y_true = np.array([])
                                     all_y_prob = np.array([])
                             
                             y0 = np.sort(all_y_prob[all_y_true==0])
                             y1 = np.sort(all_y_prob[all_y_true==1])
                             n0 = len(y0); n1 = len(y1)
                             
                             if n0 > 0:
                                 ax_ks.plot(y0, np.arange(1, n0+1)/n0, 'r', label='Neg CDF', lw=2)
                             if n1 > 0:
                                 ax_ks.plot(y1, np.arange(1, n1+1)/n1, 'b', label='Pos CDF', lw=2)
                             
                             # KS line
                             x_base = np.linspace(0, 1, 1000)
                             c0 = np.interp(x_base, y0, np.arange(1, n0+1)/n0, left=0, right=1) if n0 > 0 else np.zeros(1000)
                             c1 = np.interp(x_base, y1, np.arange(1, n1+1)/n1, left=0, right=1) if n1 > 0 else np.zeros(1000)
                             d = np.abs(c0 - c1)
                             ks_x = x_base[np.argmax(d)]
                             ax_ks.plot([ks_x, ks_x], [c1[np.argmax(d)], c0[np.argmax(d)]], 'k--', lw=2, label=f'KS={np.max(d):.3f}')
                             
                             ax_ks.set_title("CDF KS Plot")
                             ax_ks.set_xlabel("Probability")
                             ax_ks.set_ylabel("CDF")
                             ax_ks.legend()
                             ax_ks.grid(alpha=0.3)
                             
                             st.pyplot(fig_ks)
                             
                             # Save
                             buf = io.BytesIO()
                             fig_ks.savefig(buf, format='png', bbox_inches='tight')
                             buf.seek(0)
                             cv_imgs["ks_cdf"] = buf.read()
                             plt.close(fig_ks)
                             
                else: 
                     # Regression Plots - Enhanced
                     cv_imgs = {}
                     y_true_oof = np.array(oof["y_true"])
                     y_pred_oof = np.array(oof["y_pred"])
                     residuals = y_true_oof - y_pred_oof
                     
                     # Calculate R² for annotation
                     r2_val = 1 - (np.sum(residuals**2) / np.sum((y_true_oof - np.mean(y_true_oof))**2))
                     
                     # --- ROW 1: Pred vs Actual & Residuals vs Predicted ---
                     c_r1, c_r2 = st.columns(2)
                     
                     with c_r1:
                         st.write("**Predicted vs Actual (CV Pooled)**")
                         fig_pva, ax_pva = plt.subplots(figsize=(6, 6))
                         ax_pva.scatter(y_true_oof, y_pred_oof, alpha=0.4, s=20, c='steelblue')
                         # Perfect fit line
                         lims = [min(y_true_oof.min(), y_pred_oof.min()), max(y_true_oof.max(), y_pred_oof.max())]
                         ax_pva.plot(lims, lims, 'r--', lw=2, label='Perfect Fit (y=x)')
                         ax_pva.set_xlabel("Actual", fontsize=11)
                         ax_pva.set_ylabel("Predicted", fontsize=11)
                         ax_pva.set_title(f"Predicted vs Actual (R² = {r2_val:.4f})", fontsize=12, fontweight='bold')
                         ax_pva.legend()
                         ax_pva.grid(alpha=0.3)
                         st.pyplot(fig_pva)
                         # Save
                         buf = io.BytesIO()
                         fig_pva.savefig(buf, format='png', bbox_inches='tight')
                         buf.seek(0)
                         cv_imgs["pred_vs_actual"] = buf.read()
                         plt.close(fig_pva)
                     
                     with c_r2:
                         st.write("**Residuals vs Predicted**")
                         fig_rvp, ax_rvp = plt.subplots(figsize=(6, 6))
                         ax_rvp.scatter(y_pred_oof, residuals, alpha=0.4, s=20, c='darkorange')
                         ax_rvp.axhline(y=0, color='red', linestyle='--', lw=2)
                         ax_rvp.set_xlabel("Predicted", fontsize=11)
                         ax_rvp.set_ylabel("Residual (Actual - Predicted)", fontsize=11)
                         ax_rvp.set_title("Residuals vs Predicted", fontsize=12, fontweight='bold')
                         ax_rvp.grid(alpha=0.3)
                         st.pyplot(fig_rvp)
                         # Save
                         buf = io.BytesIO()
                         fig_rvp.savefig(buf, format='png', bbox_inches='tight')
                         buf.seek(0)
                         cv_imgs["residuals_vs_pred"] = buf.read()
                         plt.close(fig_rvp)
                     
                     # --- ROW 2: Residual Histogram & Q-Q Plot ---
                     c_r3, c_r4 = st.columns(2)
                     
                     with c_r3:
                         st.write("**Residual Distribution**")
                         fig_hist, ax_hist = plt.subplots(figsize=(6, 5))
                         ax_hist.hist(residuals, bins=30, edgecolor='black', alpha=0.7, color='teal')
                         ax_hist.axvline(x=0, color='red', linestyle='--', lw=2, label='Zero')
                         ax_hist.axvline(x=np.mean(residuals), color='blue', linestyle='-', lw=2, label=f'Mean={np.mean(residuals):.3f}')
                         ax_hist.set_xlabel("Residual", fontsize=11)
                         ax_hist.set_ylabel("Frequency", fontsize=11)
                         ax_hist.set_title("Residual Histogram", fontsize=12, fontweight='bold')
                         ax_hist.legend()
                         ax_hist.grid(alpha=0.3)
                         st.pyplot(fig_hist)
                         # Save
                         buf = io.BytesIO()
                         fig_hist.savefig(buf, format='png', bbox_inches='tight')
                         buf.seek(0)
                         cv_imgs["residual_hist"] = buf.read()
                         plt.close(fig_hist)
                     
                     with c_r4:
                         st.write("**Residual Q-Q Plot**")
                         fig_qq, ax_qq = plt.subplots(figsize=(6, 5))
                         scipy_stats.probplot(residuals, dist="norm", plot=ax_qq)
                         ax_qq.set_title("Q-Q Plot (Normality Check)", fontsize=12, fontweight='bold')
                         ax_qq.grid(alpha=0.3)
                         st.pyplot(fig_qq)
                         # Save
                         buf = io.BytesIO()
                         fig_qq.savefig(buf, format='png', bbox_inches='tight')
                         buf.seek(0)
                         cv_imgs["qq_plot"] = buf.read()
                         plt.close(fig_qq)
                     
                     # --- ROW 3: RMSE Distribution across folds (box plot) ---
                     if "rmse" in stats and isinstance(stats["rmse"], dict) and "raw" in stats["rmse"]:
                         st.write("**CV Metric Distribution**")
                         c_r5, c_r6 = st.columns(2)
                         
                         with c_r5:
                             fig_box, ax_box = plt.subplots(figsize=(6, 4))
                             metric_data = []
                             metric_names = []
                             for metric_key in ["rmse", "mae", "r2"]:
                                 if metric_key in stats and isinstance(stats[metric_key], dict) and "raw" in stats[metric_key]:
                                     metric_data.append(stats[metric_key]["raw"])
                                     metric_names.append(metric_key.upper())
                             if metric_data:
                                 ax_box.boxplot(metric_data, labels=metric_names)
                                 ax_box.set_ylabel("Value", fontsize=11)
                                 ax_box.set_title("CV Metric Distribution (across folds)", fontsize=12, fontweight='bold')
                                 ax_box.grid(alpha=0.3, axis='y')
                                 st.pyplot(fig_box)
                                 # Save
                                 buf = io.BytesIO()
                                 fig_box.savefig(buf, format='png', bbox_inches='tight')
                                 buf.seek(0)
                                 cv_imgs["cv_metrics_box"] = buf.read()
                                 plt.close(fig_box)

            # Update Buffer 1
            if "report_buffer" not in st.session_state: st.session_state["report_buffer"] = {}
            if "development" not in st.session_state["report_buffer"]: st.session_state["report_buffer"]["development"] = {}
            st.session_state["report_buffer"]["development"]["cv_images"] = cv_imgs

        # Final Model & Report Buffer
        st.divider()
        st.subheader("2. Final Model Evaluation")
        
        # Calculate Metrics (RESTORE FULL LIST & ORDER)
        scores_dict = {}
        if task_type == "classification":
             
             # 1. Probability Metrics (if available)
             if y_prob is not None:
                 try:
                     scores_dict["roc_auc"] = roc_auc_score(y, y_prob)
                     scores_dict["pr_auc"] = average_precision_score(y, y_prob)
                     scores_dict["brier"] = brier_score_loss(y, y_prob)
                     scores_dict["log_loss"] = log_loss(y, y_prob)
                     scores_dict["gini"] = 2 * scores_dict["roc_auc"] - 1
                     
                     p0 = y_prob[y==0]
                     p1 = y_prob[y==1]
                     scores_dict["ks"] = ks_2samp(p0, p1).statistic
                 except Exception as e:
                     st.warning(f"Could not calc prob metrics: {e}")

             # 2. Class Metrics
             scores_dict["f1"] = f1_score(y, y_pred, zero_division=0)
             scores_dict["precision"] = precision_score(y, y_pred, zero_division=0)
             scores_dict["recall"] = recall_score(y, y_pred, zero_division=0)
             scores_dict["accuracy"] = accuracy_score(y, y_pred)
             scores_dict["bal_acc"] = balanced_accuracy_score(y, y_pred)
             scores_dict["mcc"] = matthews_corrcoef(y, y_pred)

        else:
             scores_dict["rmse"] = np.sqrt(mean_squared_error(y, y_pred))
             scores_dict["mae"] = mean_absolute_error(y, y_pred)
             scores_dict["median_ae"] = median_absolute_error(y, y_pred)
             scores_dict["r2"] = r2_score(y, y_pred)
        
        # Prepare Payload
        report_payload = {
            "timestamp": res["timestamp"],
            "task_type": task_type,
            "model_config": res["config"],
            "features": features,
            "target": target,
            "train_rows": len(X),
            "metrics": scores_dict,
            "cv_metrics": cv_metrics_dict,
            "images": {}
        }
        _update_report_buffer("development", report_payload)
        
        # Tabs Final
        t_f_m, t_f_p = st.tabs(["Final Metrics", "Diagnostic Plots"])
        with t_f_m:
             st.dataframe(pd.DataFrame(list(scores_dict.items()), columns=["Metric", "Value"]))
        
        with t_f_p:
             # Final Plots (CM / Imp + Diagnostic Curves)
             dev_imgs = {}
             if task_type == "classification":
                  
                  # Row 1: Confusion Matrix & Feature Importance
                  c_d1, c_d2 = st.columns(2)
                  with c_d1:
                      cm = confusion_matrix(y, y_pred)
                      st.write("Confusion Matrix")
                      st.write(pd.DataFrame(cm))
                      # Save Img
                      fig, ax = plt.subplots(figsize=(4,3))
                      sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                      buf = io.BytesIO()
                      fig.savefig(buf, format='png', bbox_inches='tight')
                      buf.seek(0)
                      dev_imgs["confusion_matrix"] = buf.read()
                      plt.close(fig)
                  with c_d2:
                      if hasattr(model, "feature_importances_") or hasattr(model, "coef_"):
                           fi = getattr(model, "feature_importances_", getattr(model, "coef_", None))
                           if fi is not None:
                               if len(fi.shape) > 1: fi = fi[0]
                               df_imp = pd.DataFrame({"Feature": features, "Imp": fi}).sort_values("Imp", ascending=False).head(10)
                               st.bar_chart(df_imp.set_index("Feature"))
                               # Save
                               fig, ax = plt.subplots(figsize=(4,4))
                               ax.barh(df_imp["Feature"], df_imp["Imp"])
                               buf = io.BytesIO()
                               fig.savefig(buf, format='png', bbox_inches='tight')
                               buf.seek(0)
                               dev_imgs["feature_importance"] = buf.read()
                               plt.close(fig)
                  
                  # Row 2: Diagnostic Curves (RESTORED)
                  if y_prob is not None:
                       with st.expander("Diagnostic Curves (Full Model)", expanded=True):
                           
                           # Row 2a: ROC & PR
                           c_r1, c_r2 = st.columns(2)
                           with c_r1:
                               fpr, tpr, th_roc = roc_curve(y, y_prob)
                               auc_roc = roc_auc_score(y, y_prob)
                               fig_roc, ax_roc = plt.subplots(figsize=(5, 5))
                               ax_roc.plot(fpr, tpr, color='blue', lw=2, label=f'ROC (AUC = {auc_roc:.3f})')
                               ax_roc.plot([0,1], [0,1], 'r--', label='Random')
                               ax_roc.set(title="ROC Curve", xlabel="FPR", ylabel="TPR")
                               ax_roc.legend(loc='lower right')
                               ax_roc.grid(alpha=0.3)
                               st.pyplot(fig_roc)
                               # Save
                               buf = io.BytesIO()
                               fig_roc.savefig(buf, format='png', bbox_inches='tight')
                               buf.seek(0)
                               dev_imgs["roc"] = buf.read()
                               plt.close(fig_roc)
                               
                           with c_r2:
                               prec, rec, th_pr = precision_recall_curve(y, y_prob)
                               auc_pr = auc(rec, prec)
                               fig_pr, ax_pr = plt.subplots(figsize=(5, 5))
                               ax_pr.plot(rec, prec, color='green', lw=2, label=f'PR (AUC = {auc_pr:.3f})')
                               ax_pr.set(title="Precision-Recall Curve", xlabel="Recall", ylabel="Precision")
                               ax_pr.legend(loc='lower left')
                               ax_pr.grid(alpha=0.3)
                               st.pyplot(fig_pr)
                               # Save
                               buf = io.BytesIO()
                               fig_pr.savefig(buf, format='png', bbox_inches='tight')
                               buf.seek(0)
                               dev_imgs["pr"] = buf.read()
                               plt.close(fig_pr)
                           
                           # Row 2b: F1 & KS
                           c_r3, c_r4 = st.columns(2)
                           with c_r3:
                               with np.errstate(divide='ignore', invalid='ignore'):
                                   f1 = 2 * (prec * rec) / (prec + rec)
                               f1 = np.nan_to_num(f1)
                               if len(th_pr) < len(f1): f1_plot = f1[:len(th_pr)]
                               else: f1_plot = f1
                               
                               # Find max F1 and best threshold
                               max_f1_idx = np.argmax(f1_plot)
                               max_f1 = f1_plot[max_f1_idx]
                               best_thresh = th_pr[max_f1_idx] if max_f1_idx < len(th_pr) else 0.5
                               
                               fig_f1, ax_f1 = plt.subplots(figsize=(5, 5))
                               ax_f1.plot(th_pr, f1_plot, color='purple', lw=2, label=f'F1 Score')
                               ax_f1.axvline(x=best_thresh, color='red', linestyle='--', lw=1.5, label=f'Best Threshold = {best_thresh:.3f}')
                               ax_f1.axhline(y=max_f1, color='orange', linestyle=':', lw=1.5, label=f'Max F1 = {max_f1:.3f}')
                               ax_f1.scatter([best_thresh], [max_f1], color='red', s=80, zorder=5)
                               ax_f1.set(title="F1 vs Threshold", xlabel="Threshold", ylabel="F1 Score")
                               ax_f1.legend(loc='lower right', fontsize=9)
                               ax_f1.grid(alpha=0.3)
                               st.pyplot(fig_f1)
                               # Save
                               buf = io.BytesIO()
                               fig_f1.savefig(buf, format='png', bbox_inches='tight')
                               buf.seek(0)
                               dev_imgs["f1"] = buf.read()
                               plt.close(fig_f1)
                           
                           with c_r4:
                               # CDF KS Plot
                               y0 = np.sort(y_prob[y==0])
                               y1 = np.sort(y_prob[y==1])
                               n0 = len(y0); n1 = len(y1)
                               
                               fig_ks, ax_ks = plt.subplots(figsize=(5, 5))
                               if n0 > 0:
                                   ax_ks.plot(y0, np.arange(1, n0+1)/n0, 'r', label='Neg CDF', lw=2)
                               if n1 > 0:
                                   ax_ks.plot(y1, np.arange(1, n1+1)/n1, 'b', label='Pos CDF', lw=2)
                               
                               # KS line
                               x_base = np.linspace(0, 1, 1000)
                               c0 = np.interp(x_base, y0, np.arange(1, n0+1)/n0, left=0, right=1) if n0 > 0 else np.zeros(1000)
                               c1 = np.interp(x_base, y1, np.arange(1, n1+1)/n1, left=0, right=1) if n1 > 0 else np.zeros(1000)
                               d = np.abs(c0 - c1)
                               ks_x = x_base[np.argmax(d)]
                               ax_ks.plot([ks_x, ks_x], [c1[np.argmax(d)], c0[np.argmax(d)]], 'k--', lw=2, label=f'KS={np.max(d):.3f}')
                               
                               ax_ks.set(title="CDF KS Plot", xlabel="Probability", ylabel="CDF")
                               ax_ks.legend()
                               ax_ks.grid(alpha=0.3)
                               st.pyplot(fig_ks)
                               # Save
                               buf = io.BytesIO()
                               fig_ks.savefig(buf, format='png', bbox_inches='tight')
                               buf.seek(0)
                               dev_imgs["ks_cdf"] = buf.read()
                               plt.close(fig_ks)
             else:
                  # REGRESSION Full Model Diagnostic Plots
                  dev_imgs = {}
                  
                  residuals = y.values - y_pred if hasattr(y, 'values') else np.array(y) - y_pred
                  r2_val = scores_dict.get("r2", 0)
                  
                  with st.expander("Regression Diagnostics (Full Model)", expanded=True):
                       # Row 1: Pred vs Actual & Residuals vs Predicted
                       c_d1, c_d2 = st.columns(2)
                       
                       with c_d1:
                           st.write("**Predicted vs Actual**")
                           fig_pva, ax_pva = plt.subplots(figsize=(6, 6))
                           ax_pva.scatter(y, y_pred, alpha=0.4, s=20, c='steelblue')
                           lims = [min(np.min(y), np.min(y_pred)), max(np.max(y), np.max(y_pred))]
                           ax_pva.plot(lims, lims, 'r--', lw=2, label='Perfect Fit (y=x)')
                           ax_pva.set_xlabel("Actual", fontsize=11)
                           ax_pva.set_ylabel("Predicted", fontsize=11)
                           ax_pva.set_title(f"Predicted vs Actual (R² = {r2_val:.4f})", fontsize=12, fontweight='bold')
                           ax_pva.legend()
                           ax_pva.grid(alpha=0.3)
                           st.pyplot(fig_pva)
                           # Save
                           buf = io.BytesIO()
                           fig_pva.savefig(buf, format='png', bbox_inches='tight')
                           buf.seek(0)
                           dev_imgs["pred_vs_actual"] = buf.read()
                           plt.close(fig_pva)
                       
                       with c_d2:
                           st.write("**Residuals vs Predicted**")
                           fig_rvp, ax_rvp = plt.subplots(figsize=(6, 6))
                           ax_rvp.scatter(y_pred, residuals, alpha=0.4, s=20, c='darkorange')
                           ax_rvp.axhline(y=0, color='red', linestyle='--', lw=2)
                           ax_rvp.set_xlabel("Predicted", fontsize=11)
                           ax_rvp.set_ylabel("Residual (Actual - Predicted)", fontsize=11)
                           ax_rvp.set_title("Residuals vs Predicted", fontsize=12, fontweight='bold')
                           ax_rvp.grid(alpha=0.3)
                           st.pyplot(fig_rvp)
                           # Save
                           buf = io.BytesIO()
                           fig_rvp.savefig(buf, format='png', bbox_inches='tight')
                           buf.seek(0)
                           dev_imgs["residuals_vs_pred"] = buf.read()
                           plt.close(fig_rvp)
                       
                       # Row 2: Residual Histogram & Q-Q Plot
                       c_d3, c_d4 = st.columns(2)
                       
                       with c_d3:
                           st.write("**Residual Distribution**")
                           fig_hist, ax_hist = plt.subplots(figsize=(6, 5))
                           ax_hist.hist(residuals, bins=30, edgecolor='black', alpha=0.7, color='teal')
                           ax_hist.axvline(x=0, color='red', linestyle='--', lw=2, label='Zero')
                           ax_hist.axvline(x=np.mean(residuals), color='blue', linestyle='-', lw=2, label=f'Mean={np.mean(residuals):.3f}')
                           ax_hist.set_xlabel("Residual", fontsize=11)
                           ax_hist.set_ylabel("Frequency", fontsize=11)
                           ax_hist.set_title("Residual Histogram", fontsize=12, fontweight='bold')
                           ax_hist.legend()
                           ax_hist.grid(alpha=0.3)
                           st.pyplot(fig_hist)
                           # Save
                           buf = io.BytesIO()
                           fig_hist.savefig(buf, format='png', bbox_inches='tight')
                           buf.seek(0)
                           dev_imgs["residual_hist"] = buf.read()
                           plt.close(fig_hist)
                       
                       with c_d4:
                           st.write("**Residual Q-Q Plot**")
                           fig_qq, ax_qq = plt.subplots(figsize=(6, 5))
                           scipy_stats.probplot(residuals, dist="norm", plot=ax_qq)
                           ax_qq.set_title("Q-Q Plot (Normality Check)", fontsize=12, fontweight='bold')
                           ax_qq.grid(alpha=0.3)
                           st.pyplot(fig_qq)
                           # Save
                           buf = io.BytesIO()
                           fig_qq.savefig(buf, format='png', bbox_inches='tight')
                           buf.seek(0)
                           dev_imgs["qq_plot"] = buf.read()
                           plt.close(fig_qq)
                       
                       # Row 3: Feature Importance (if available)
                       if hasattr(model, "feature_importances_") or hasattr(model, "coef_"):
                           c_d5, _ = st.columns(2)
                           with c_d5:
                               st.write("**Feature Importance**")
                               fi = getattr(model, "feature_importances_", getattr(model, "coef_", None))
                               if fi is not None:
                                   if len(fi.shape) > 1: fi = fi[0]
                                   df_imp = pd.DataFrame({"Feature": features, "Importance": np.abs(fi)}).sort_values("Importance", ascending=False).head(10)
                                   fig_imp, ax_imp = plt.subplots(figsize=(6, 4))
                                   ax_imp.barh(df_imp["Feature"][::-1], df_imp["Importance"][::-1], color='mediumseagreen')
                                   ax_imp.set_xlabel("Importance", fontsize=11)
                                   ax_imp.set_title("Top 10 Feature Importance", fontsize=12, fontweight='bold')
                                   ax_imp.grid(alpha=0.3, axis='x')
                                   st.pyplot(fig_imp)
                                   # Save
                                   buf = io.BytesIO()
                                   fig_imp.savefig(buf, format='png', bbox_inches='tight')
                                   buf.seek(0)
                                   dev_imgs["feature_importance"] = buf.read()
                                   plt.close(fig_imp)
             
             # Save to buffer
             if "report_buffer" in st.session_state and "development" in st.session_state["report_buffer"]:
                 st.session_state["report_buffer"]["development"]["images"] = dev_imgs

    # --- REPORT BUTTON ---
    st.divider()
    if "dev_results" in st.session_state:
        st.subheader("Report")
        if st.button("Generate Development Report 📄"):
            # Use buffer
            buf = st.session_state.get("report_buffer", {}).get("development", {})
            try:
                docx_bytes = _generate_dev_report_docx(buf)
                st.download_button("Download DOCX", docx_bytes, "dev_report.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
            except Exception as e:
                st.error(f"Report Gen Failed: {e}")
