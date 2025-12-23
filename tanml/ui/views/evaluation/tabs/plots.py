# tanml/ui/pages/evaluation/tabs/plots.py
"""
Diagnostic Plots Tab - Visualizes model performance logic.
"""

import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import scipy.stats as scipy_stats
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, roc_auc_score

from tanml.ui.views.evaluation.tabs import register_tab


@register_tab(name="Diagnostic Plots Comparison", order=20, key="tab_plots")
def render(context):
    """Render the diagnostic plots tab."""
    
    st.markdown("### Model Diagnostic Plots")
    
    eval_imgs = {} # Init capture dict
    c_p1, c_p2 = st.columns(2)
    
    def _plot_diagnostics(col, title, y_true, y_pred, X_in, img_dict, prefix):
        col.write(f"**{title}**")
        
        if context.task_type == "classification":
            cm = confusion_matrix(y_true, y_pred)
            col.write("Confusion Matrix:")
            col.write(pd.DataFrame(cm, columns=["Pred 0", "Pred 1"], index=["True 0", "True 1"]))
            col.divider()
            
            if hasattr(context.model, "predict_proba"):
                try:
                    y_prob = context.model.predict_proba(X_in)[:, 1]
                    
                    with col.expander("Diagnostic Curves", expanded=True):
                        sub_c1, sub_c2 = col.columns(2)
                        
                        # 1. ROC
                        fpr, tpr, th_roc = roc_curve(y_true, y_prob)
                        fig_roc, ax_roc = plt.subplots(figsize=(4, 4))
                        ax_roc.plot(fpr, tpr, color='blue', lw=2, label=f'AUC={roc_auc_score(y_true, y_prob):.3f}')
                        ax_roc.plot([0,1], [0,1], 'r--')
                        ax_roc.set(title="ROC", xlabel="FPR", ylabel="TPR")
                        ax_roc.legend(loc="lower right")
                        sub_c1.pyplot(fig_roc)
                        _save_plot(fig_roc, img_dict, f"{prefix}_roc")
                        plt.close(fig_roc)
                        
                        # 2. PR
                        prec, rec, th_pr = precision_recall_curve(y_true, y_prob)
                        fig_pr, ax_pr = plt.subplots(figsize=(4, 4))
                        ax_pr.plot(rec, prec, color='green', lw=2)
                        ax_pr.set(title="PR Curve", xlabel="Recall", ylabel="Precision")
                        sub_c2.pyplot(fig_pr)
                        _save_plot(fig_pr, img_dict, f"{prefix}_pr")
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
                        _save_plot(fig_f1, img_dict, f"{prefix}_f1")
                        plt.close(fig_f1)
                        
                        # Classic CDF - KS Plot
                        fig_cdf = _plot_cdf_ks_local(y_true, y_prob)
                        sub_c4.pyplot(fig_cdf)
                        _save_plot(fig_cdf, img_dict, f"{prefix}_ks_cdf")
                        plt.close(fig_cdf)

                except Exception as e:
                    col.error(f"Error plotting curves: {e}")
        else:
            # Regression Plots
            residuals = y_true - y_pred
            r2_val = 1 - (np.sum(residuals**2) / np.sum((y_true - np.mean(y_true))**2)) if len(y_true) > 1 else 0
            
            # Sampling for heavy plots
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
                    _save_plot(fig_pva, img_dict, f"{prefix}_pred_actual")
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
                    _save_plot(fig_rvp, img_dict, f"{prefix}_residuals")
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
                    _save_plot(fig_hist, img_dict, f"{prefix}_residual_hist")
                    plt.close(fig_hist)
                
                with sub_c4:
                    fig_qq, ax_qq = plt.subplots(figsize=(4, 3.5))
                    scipy_stats.probplot(residuals, dist="norm", plot=ax_qq)
                    ax_qq.set_title("Q-Q Plot (Normality)", fontsize=10, fontweight='bold')
                    ax_qq.grid(alpha=0.3)
                    st.pyplot(fig_qq)
                    _save_plot(fig_qq, img_dict, f"{prefix}_qq")
                    plt.close(fig_qq)

    # Render Side by Side
    _plot_diagnostics(c_p1, "TRAIN Diagnostics", context.y_train, context.y_pred_train, context.X_train, eval_imgs, "train")
    _plot_diagnostics(c_p2, "TEST Diagnostics", context.y_test, context.y_pred_test, context.X_test, eval_imgs, "test")
    
    # Store images in context for report
    context.images.update(eval_imgs)


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


def _save_plot(fig, img_dict, key):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_dict[key] = buf.read()
