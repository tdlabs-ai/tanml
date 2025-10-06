from __future__ import annotations
import os, math
from dataclasses import dataclass, asdict
from typing import Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix, precision_recall_fscore_support,
    accuracy_score, average_precision_score, brier_score_loss, precision_recall_curve
)
from sklearn.calibration import calibration_curve

# ---------- utilities ----------
def _ensure_dir(d: str) -> str:
    os.makedirs(d, exist_ok=True)
    return d

def _savefig(path: str) -> str:
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", dpi=160)
    plt.close()
    return path

def _gini_from_auc(auc: float) -> float:
    return 2 * auc - 1 if (auc is not None and not np.isnan(auc)) else np.nan

def _ks_from_roc(fpr, tpr) -> float:
    return float(np.max(np.abs(tpr - fpr))) if len(fpr) else np.nan

def _decile_lift_table(y_true: np.ndarray, y_score: np.ndarray, pos_label: int = 1, n_bins: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"y": (y_true == pos_label).astype(int), "score": y_score})
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df["decile"] = pd.qcut(df.index + 1, q=n_bins, labels=list(range(1, n_bins + 1)))
    g = df.groupby("decile", observed=True).agg(
        total=("y", "size"),
        events=("y", "sum"),
        avg_score=("score", "mean"),
    ).reset_index()
    g["event_rate"] = g["events"] / g["total"]
    overall_rate = df["y"].mean() if df["y"].size else np.nan
    g["lift"] = g["event_rate"] / overall_rate if (overall_rate and not math.isclose(overall_rate, 0.0)) else np.nan

    # cumulative capture & gain
    g["cum_events"] = g["events"].cumsum()
    g["cum_total"] = g["total"].cumsum()
    total_events = g["events"].sum()
    g["cum_capture_rate"] = g["cum_events"] / total_events if total_events > 0 else np.nan
    g["cum_population"] = g["cum_total"] / g["total"].sum()
    g["cum_gain"] = g["cum_capture_rate"]  # same as cumulative gains curve
    return g

def _ks_curve_frame(y_true: np.ndarray, y_score: np.ndarray, pos_label: int = 1) -> pd.DataFrame:
    """
    Returns a dataframe with columns:
      population (fraction 0..1), cum_event, cum_non_event, ks_gap
    sorted by score DESC, which is standard for risk ranking.
    """
    df = pd.DataFrame({"y": (y_true == pos_label).astype(int), "score": y_score})
    if df.empty:
        return pd.DataFrame(columns=["population", "cum_event", "cum_non_event", "ks_gap"])

    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    n = len(df)

    # counts
    total_events = df["y"].sum()
    total_non_events = n - total_events

    # avoid divide-by-zero; if all one class, return empty (plotter will handle)
    if total_events == 0 or total_non_events == 0:
        return pd.DataFrame(columns=["population", "cum_event", "cum_non_event", "ks_gap"])

    cum_events = np.cumsum(df["y"].values) / total_events
    cum_non_events = np.cumsum(1 - df["y"].values) / total_non_events
    population = (np.arange(1, n + 1)) / n
    ks_gap = np.abs(cum_events - cum_non_events)

    return pd.DataFrame({
        "population": population,
        "cum_event": cum_events,
        "cum_non_event": cum_non_events,
        "ks_gap": ks_gap
    })

@dataclass
class ClassificationSummary:
    auc: float
    ks: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    pr_auc: float
    brier: float
    gini: float
    # paths
    roc_png: str
    pr_png: str
    lift_png: str
    calib_png: str
    cm_png: str
    ks_png: str        
    # tables
    confusion_csv: str
    lift_csv: str


def compute_classification_report(
    *,
    y_true: np.ndarray,
    y_score: np.ndarray,
    y_pred: np.ndarray,
    outdir: str,
    pos_label: int = 1,
    title_prefix: str = "Model"
) -> Dict[str, Any]:
    """
    Computes metrics + saves plots/CSVs for classification.
    Returns a dict ready for ReportBuilder/Jinja.
    """
    _ensure_dir(outdir)

    # --- metrics
    has_posneg = len(np.unique(y_true)) > 1
    auc = roc_auc_score(y_true, y_score) if has_posneg else np.nan
    fpr, tpr, _ = roc_curve(y_true, y_score) if has_posneg else (np.array([]), np.array([]), None)
    ks = _ks_from_roc(fpr, tpr)
    pr_auc = average_precision_score(y_true, y_score) if has_posneg else np.nan
    brier = brier_score_loss(y_true, y_score)
    gini = _gini_from_auc(auc)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=pos_label, zero_division=0
    )
    acc = accuracy_score(y_true, y_pred)

    # --- confusion matrix & CSV
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_df = pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"])
    cm_csv = os.path.join(outdir, "confusion_matrix.csv")
    cm_df.to_csv(cm_csv, index=True)

    # --- decile lift table & CSV (rounded to 2 decimals)
    lift_df = _decile_lift_table(y_true, y_score, pos_label=pos_label, n_bins=10)
    lift_df_round = lift_df.copy()
    num_cols = lift_df_round.select_dtypes(include=[np.number]).columns
    lift_df_round[num_cols] = lift_df_round[num_cols].round(2)
    lift_csv = os.path.join(outdir, "lift_table_deciles.csv")
    lift_df_round.to_csv(lift_csv, index=False)

    # --- plots
    # ROC
    if len(fpr):
        plt.figure()
        plt.plot(fpr, tpr, label=f"ROC (AUC={auc:.3f})")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.title(f"{title_prefix}: ROC Curve")
        plt.legend(loc="lower right")
    roc_png = os.path.join(outdir, "roc_curve.png")
    _savefig(roc_png)

    # PR
    pr, rc, _ = precision_recall_curve(y_true, y_score)
    plt.figure()
    plt.plot(rc, pr, label=f"PR (AP={pr_auc:.3f})")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"{title_prefix}: Precision–Recall Curve")
    plt.legend(loc="lower left")
    pr_png = os.path.join(outdir, "pr_curve.png")
    _savefig(pr_png)

    # Calibration (Reliability) curve
    prob_true, prob_pred = calibration_curve(y_true, y_score, n_bins=10, strategy="uniform")
    plt.figure()
    plt.plot(prob_pred, prob_true, marker="o", label="Reliability")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect")
    plt.xlabel("Predicted probability"); plt.ylabel("Observed frequency"); plt.title(f"{title_prefix}: Calibration")
    plt.legend(loc="upper left")
    calib_png = os.path.join(outdir, "calibration_curve.png")
    _savefig(calib_png)

    # Lift / Gain chart (use unrounded frame for smooth curve)
    plt.figure()
    plt.plot(lift_df["cum_population"], lift_df["cum_gain"], marker="o", label="Cumulative Gain")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Baseline")
    plt.xlabel("Cumulative Population"); plt.ylabel("Cumulative Gain"); plt.title(f"{title_prefix}: Cumulative Gain")
    plt.legend(loc="lower right")
    lift_png = os.path.join(outdir, "lift_gain_curve.png")
    _savefig(lift_png)

    # Confusion heatmap
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"{title_prefix}: Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Pred 0", "Pred 1"])
    plt.yticks(tick_marks, ["Actual 0", "Actual 1"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.ylabel("Actual"); plt.xlabel("Predicted")
    cm_png = os.path.join(outdir, "confusion_matrix.png")
    _savefig(cm_png)

    # --- KS curve (cumulative event vs non-event by population)
    ks_df  = _ks_curve_frame(y_true, y_score, pos_label=pos_label)
    ks_csv = os.path.join(outdir, "ks_curve.csv")
    ks_png = os.path.join(outdir, "ks_curve.png")

    if not ks_df.empty:
        # locate max KS point
        ks_idx       = int(ks_df["ks_gap"].values.argmax())
        ks_x         = float(ks_df["population"].iloc[ks_idx])
        ks_y_event   = float(ks_df["cum_event"].iloc[ks_idx])
        ks_y_nonevent= float(ks_df["cum_non_event"].iloc[ks_idx])
        ks_val_annot = abs(ks_y_event - ks_y_nonevent)

        # plot with explicit figure/axes only once
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(ks_df["population"], ks_df["cum_event"],     label="Cumulative Event")
        ax.plot(ks_df["population"], ks_df["cum_non_event"], label="Cumulative Non-Event")

        # vertical line & markers at max KS
        ax.axvline(ks_x, linestyle="--", alpha=0.7)
        ax.scatter([ks_x], [ks_y_event],    s=25)
        ax.scatter([ks_x], [ks_y_nonevent], s=25)

        # readable annotation (two lines, boxed)
        ax.annotate(
            f"KS = {ks_val_annot:.1%}\nPop = {ks_x:.1%}",
            xy=(ks_x, (ks_y_event + ks_y_nonevent) / 2.0),
            xytext=(ks_x + 0.05, min(0.9, (ks_y_event + ks_y_nonevent) / 2.0 + 0.1)),
            arrowprops=dict(arrowstyle="->", color="black", lw=1),
            ha="left", va="center", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=0.8),
        )

        ax.set_xlabel("Population (fraction)")
        ax.set_ylabel("Cumulative share")
        ax.set_title(f"{title_prefix}: KS Curve")
        ax.legend(loc="lower right")

        ks_df.to_csv(ks_csv, index=False)
        _savefig(ks_png)
    else:
        # Write header-only CSV and a placeholder figure so ks_png always exists
        pd.DataFrame(columns=["population","cum_event","cum_non_event","ks_gap"]).to_csv(ks_csv, index=False)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_title(f"{title_prefix}: KS Curve (not available)")
        _savefig(ks_png)

    # --- build summaries: raw + rounded for display
    summary = ClassificationSummary(
        auc=float(auc) if auc == auc else np.nan,  # handle NaN
        ks=float(ks) if ks == ks else np.nan,
        accuracy=float(acc),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        pr_auc=float(pr_auc) if pr_auc == pr_auc else np.nan,
        brier=float(brier),
        gini=float(gini) if gini == gini else np.nan,
        roc_png=roc_png,
        pr_png=pr_png,
        lift_png=lift_png,
        calib_png=calib_png,
        cm_png=cm_png,
        ks_png=ks_png,                # <---- NEW
        confusion_csv=cm_csv,
        lift_csv=lift_csv,
    )
    summary_raw = asdict(summary)

    # round only numeric metric fields to 2 decimals; keep paths as-is
    metric_fields = {"auc", "ks", "accuracy", "precision", "recall", "f1", "pr_auc", "brier", "gini"}
    summary_rounded = {}
    for k, v in summary_raw.items():
        if k in metric_fields:
            try:
                summary_rounded[k] = None if v is None or (isinstance(v, float) and np.isnan(v)) else round(float(v), 2)
            except Exception:
                summary_rounded[k] = v
        else:
            summary_rounded[k] = v

    return {
        "summary": summary_rounded,   # rounded for report/UI display
        "summary_raw": summary_raw,   # full precision preserved for rules/debug
        "tables": {
            "confusion_csv": cm_csv,
            "lift_csv": lift_csv,
            "ks_csv": ks_csv,       
        },
        "plots": {
            "roc": roc_png,
            "pr": pr_png,
            "lift": lift_png,
            "calibration": calib_png,
            "confusion": cm_png,
            "ks": ks_png,          
        },
        # return rounded rows so the DOCX table looks clean
        "deciles": lift_df_round.to_dict(orient="records"),
    }
