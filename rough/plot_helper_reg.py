
import numpy as np
import matplotlib.pyplot as plt

try:
    from scipy import stats as _scipy_stats
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

def _residuals(y_true, y_pred):
    return np.array(y_true) - np.array(y_pred)

def plot_reg_residuals_vs_pred(y_true, y_pred, save_path):
    resid = _residuals(y_true, y_pred)
    plt.figure()
    plt.scatter(y_pred, resid, s=12, alpha=0.75)
    plt.axhline(0.0, color='r', linestyle='--')
    plt.xlabel("Predicted")
    plt.ylabel("Residual (y_true - y_pred)")
    plt.title("Residuals vs Predicted")
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()

def plot_reg_residual_hist(y_true, y_pred, save_path):
    resid = _residuals(y_true, y_pred)
    plt.figure()
    plt.hist(resid, bins=30, alpha=0.9, color='steelblue', edgecolor='black')
    plt.xlabel("Residual")
    plt.ylabel("Count")
    plt.title("Residual Distribution")
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()

def plot_reg_qq(y_true, y_pred, save_path):
    if not _HAS_SCIPY:
        # Create a placeholder image containing text
        plt.figure(figsize=(5, 3))
        plt.text(0.5, 0.5, "Q-Q Plot unavailable\n(SciPy not installed)", 
                 ha='center', va='center')
        plt.axis('off')
        plt.savefig(save_path, dpi=160)
        plt.close()
        return

    resid = _residuals(y_true, y_pred)
    osm, osr = _scipy_stats.probplot(resid, dist="norm", fit=False)
    plt.figure()
    plt.scatter(osm, osr, s=12, alpha=0.8)
    
    # Reference line
    mn = float(min(np.min(osm), np.min(osr)))
    mx = float(max(np.max(osm), np.max(osr)))
    plt.plot([mn, mx], [mn, mx], color='r', linestyle='--')
    
    plt.xlabel("Theoretical Quantiles (Normal)")
    plt.ylabel("Ordered Residuals")
    plt.title("Residuals Q–Q Plot")
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()

def plot_reg_abs_error_box(y_true, y_pred, save_path):
    resid = _residuals(y_true, y_pred)
    abs_err = np.abs(resid)
    plt.figure()
    plt.boxplot(abs_err, vert=True, showfliers=True)
    plt.ylabel("|Residual|")
    plt.title("Absolute Error — Box Plot")
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()

def plot_reg_abs_error_violin(y_true, y_pred, save_path):
    resid = _residuals(y_true, y_pred)
    abs_err = np.abs(resid)
    plt.figure()
    plt.violinplot(abs_err, showmeans=True, showmedians=True)
    plt.ylabel("|Residual|")
    plt.title("Absolute Error — Violin Plot")
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()
