# tanml/report/report_builder.py
from docx import Document
from docx.shared import Inches, Mm
from pathlib import Path
import os, re, math, copy as pycopy
from importlib.resources import files
import numpy as np  # needed for rounding helpers etc.
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_ALIGN_VERTICAL



TMP_DIR = Path(__file__).resolve().parents[1] / "tmp_report_assets"
TMP_DIR.mkdir(exist_ok=True)

class AttrDict(dict):
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(item)

def _read_csv_as_rows(path, max_rows=None):
    import csv
    rows = []
    if not path or not os.path.exists(path):
        return rows
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for i, row in enumerate(r):
            if max_rows is not None and i >= max_rows:
                break
            rows.append(row)
    return rows


USE_TEXT_PLACEHOLDERS = True
TXT = {
    "not_applicable": "Not applicable",
    "not_provided": "Dataset not provided",
    "none_detected": "None detected",
    "no_issues": "None (no issues detected)",
    "unknown": "Unknown",
    "dash": "—",
}
def _p(key: str) -> str:
    return TXT["dash"] if not USE_TEXT_PLACEHOLDERS else TXT.get(key, TXT["unknown"])


def _fallback_pairs_from_corr_matrix(corr_csv_path: str, threshold: float, top_k: int = 200):
    """
    Build 'top pairs' rows from a saved correlation MATRIX CSV when the engine
    did not emit 'top_pairs_main_csv'/'top_pairs_csv'.

    Returns a list of dicts with headers:
      ["feature_i", "feature_j", "corr", "n_used", "pct_missing_i", "pct_missing_j"]
    """
    rows = []
    if not corr_csv_path or not os.path.exists(corr_csv_path):
        return rows
    try:
        import pandas as pd
        df = pd.read_csv(corr_csv_path, index_col=0)
        # Ensure square matrix with aligned labels
        if df.shape[0] == 0 or df.shape[0] != df.shape[1]:
            return rows
        cols = list(df.columns)

        # Collect upper triangle |r| >= threshold
        pairs = []
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                try:
                    r = float(df.iloc[i, j])
                except Exception:
                    continue
                if abs(r) >= float(threshold):
                    pairs.append((cols[i], cols[j], r))

        # Sort by absolute correlation desc, then take top_k
        pairs.sort(key=lambda t: abs(t[2]), reverse=True)
        pairs = pairs[: int(top_k)]

        # Map to expected schema; we don't have n_used / pct_missing_* here
        for a, b, r in pairs:
            rows.append({
                "feature_i": a,
                "feature_j": b,
                "corr": f"{float(r):.4f}",
                "n_used": "",               # unknown in matrix-only fallback
                "pct_missing_i": "",        # unknown
                "pct_missing_j": "",        # unknown
            })
        return rows
    except Exception:
        return []

def build_table(doc, headers, rows):
    tbl = doc.add_table(rows=1, cols=len(headers))
    tbl.style = "Table Grid"
    for i, h in enumerate(headers):
        tbl.rows[0].cells[i].text = str(h)
    for r in rows:
        vals = [r.get(h, "") for h in headers] if isinstance(r, dict) else list(r)
        vals += [""] * (len(headers) - len(vals))
        row = tbl.add_row().cells
        for i, v in enumerate(vals):
            row[i].text = str(v)
    return tbl

# --- Pretty Decile Lift Table (3-line headers, auto-fit, numeric alignment) ---
def build_decile_lift_table(doc, headers, rows):
    """
    Renders the decile lift table with:
      - compact labels
      - up to THREE lines per header cell (forced line breaks)
      - widths auto-scaled to page text width
      - right-aligned numeric columns
    """

    # 1) Compact labels (keep Total)
    label_map = {
        "decile": "Decile",
        "total": "Total",
        "events": "Events",
        "avg_score": "Avg score",
        "event_rate": "Event rate",
        "lift": "Lift",
        "cum_events": "Cum. events",
        "cum_total": "Cum. total",
        "cum_capture_rate": "Cum. capture rate",
        "cum_population": "Cum. population",
        "cum_gain": "Cum. gain",
    }
    pretty_headers = [label_map.get(h.strip(), h.replace("_", " ").strip()) for h in headers]

    # 2) Helper: force up to 3 lines per label (balanced by words)
    def split_to_max_lines(label: str, max_lines: int = 3):
        parts = str(label).split()
        if len(parts) <= 1:
            return [label]
        # Greedy balance into up to max_lines buckets
        buckets = [[] for _ in range(max_lines)]
        # pre-fill minimal distribution
        for i, w in enumerate(parts):
            buckets[i % max_lines].append(w)
        # join each bucket; trim empty
        lines = [" ".join(b).strip() for b in buckets if b]
        # remove trailing empties
        lines = [ln for ln in lines if ln]
        return lines[:max_lines]

    # 3) Build table (single header row; we’ll insert line breaks in each header cell)
    tbl = doc.add_table(rows=1, cols=len(pretty_headers))
    tbl.style = "Table Grid"
    tbl.autofit = False

    # 4) Header row with forced breaks (up to 3 lines)
    hdr_row = tbl.rows[0]
    for j, h in enumerate(pretty_headers):
        cell = hdr_row.cells[j]
        # Clear cell safely
        cell.text = ""
        p = cell.paragraphs[0] if cell.paragraphs else cell.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER

        lines = split_to_max_lines(h, max_lines=3)
        if not lines:
            continue
        run = p.add_run(str(lines[0]))
        for seg in lines[1:]:
            run.add_break()            # hard line break
            p.add_run(str(seg))

    # 5) Body rows (preserve original header order)
    for r in rows:
        vals = [r.get(h, "") for h in headers] if isinstance(r, dict) else list(r)
        vals += [""] * (len(pretty_headers) - len(vals))
        cells = tbl.add_row().cells
        for j, v in enumerate(vals):
            cells[j].text = "" if v is None else str(v)

    # 6) Base widths (tight) in inches — will be scaled to fit page text width
    #    Order corresponds to incoming headers: decile, total, events, ...
    base_widths = [0.50, 0.60, 0.60, 0.70, 0.70, 0.55, 0.80, 0.80, 0.95, 0.95, 0.85]
    widths = base_widths[:len(pretty_headers)]

    # 7) Compute usable width from document section (with a small safety margin)
    try:
        sec = doc.sections[0]
        usable_in = float(sec.page_width - sec.left_margin - sec.right_margin) / 914400.0
        usable_in = max(usable_in - 0.05, 5.8)
    except Exception:
        usable_in = 6.7

    # 8) Auto-scale to fit
    total_w = sum(widths)
    if total_w > 0 and usable_in > 0:
        scale = min(1.0, usable_in / total_w)
        widths = [w * scale for w in widths]

    # 9) Apply widths
    for j, w in enumerate(widths):
        for row in tbl.rows:
            row.cells[j].width = Inches(w)

    # 10) Right-align numeric columns
    numeric_like = {
        "Total", "Events", "Avg score", "Event rate", "Lift",
        "Cum. events", "Cum. total", "Cum. capture rate",
        "Cum. population", "Cum. gain",
    }
    for i, h in enumerate(pretty_headers):
        right = h in numeric_like
        for row in tbl.rows[1:]:
            for p in row.cells[i].paragraphs:
                p.alignment = WD_ALIGN_PARAGRAPH.RIGHT if right else WD_ALIGN_PARAGRAPH.LEFT

    return tbl



def insert_after(doc, anchor, tbl):
    for p in doc.paragraphs:
        if anchor.lower() in p.text.lower():
            parent = p._p.getparent()
            parent.insert(parent.index(p._p) + 1, tbl._tbl)
            return
    print(f"⚠️ anchor «{anchor}» not found")

def insert_image_grid(doc, anchor: str, img_paths, cols: int = 3, width_in: float = 2.2):
    paths = [p for p in (img_paths or []) if p and os.path.exists(p)]
    if not paths:
        for p in doc.paragraphs:
            if anchor.lower() in p.text.lower():
                after = p.insert_paragraph_after()
                after.add_run("(no plots available)")
                return
        print(f"⚠️ anchor «{anchor}» not found (no plots)")
        return

    rows = max(1, math.ceil(len(paths) / max(1, cols)))
    tbl = doc.add_table(rows=rows, cols=cols)
    tbl.style = "Table Grid"
    i = 0
    for r in range(rows):
        for c in range(cols):
            cell = tbl.cell(r, c)
            if i < len(paths):
                para = cell.paragraphs[0]
                para.text = ""
                run = para.add_run()
                run.add_picture(paths[i], width=Inches(width_in))
                i += 1
            else:
                cell.text = ""
    insert_after(doc, anchor, tbl)

# ---------- placeholder & image replacement ----------
_PLACEHOLDER = re.compile(r"\{\{([A-Za-z0-9_\.]+)\}\}")
_IMG_MARKER  = re.compile(r"\[\[IMG:([A-Za-z0-9_\-]+)\]\]")

def _get_nested(d, dotted, default=""):
    cur = d
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur

def _replace_text_placeholders(doc, mapping):
    # paragraphs
    for p in doc.paragraphs:
        full = p.text
        def repl(m):
            key = m.group(1)
            return str(mapping.get(key, _get_nested(mapping, key, "")) or "")
        new_full = _PLACEHOLDER.sub(repl, full)
        if new_full != full:
            for _ in range(len(p.runs)-1, -1, -1):
                p.runs[_].text = ""
            if p.runs:
                p.runs[0].text = new_full
            else:
                p.add_run(new_full)
    # table cells
    for t in doc.tables:
        for row in t.rows:
            for cell in row.cells:
                for p in cell.paragraphs:
                    full = p.text
                    new_full = _PLACEHOLDER.sub(
                        lambda m: str(mapping.get(m.group(1), _get_nested(mapping, m.group(1), "")) or ""),
                        full
                    )
                    if new_full != full:
                        for _ in range(len(p.runs)-1, -1, -1):
                            p.runs[_].text = ""
                        if p.runs:
                            p.runs[0].text = new_full
                        else:
                            p.add_run(new_full)

def _replace_image_markers(doc, images_map, *, width_in=4.8):
    def _put_image_in_paragraph(p, path):
        if path and os.path.exists(path):
            p.text = ""
            run = p.add_run()
            run.add_picture(path, width=Inches(width_in))
        else:
            p.text = "(image not available)"
    # paragraphs
    for p in list(doc.paragraphs):
        m = _IMG_MARKER.search(p.text)
        if m:
            key = m.group(1)
            _put_image_in_paragraph(p, images_map.get(key))
    # table cells
    for t in doc.tables:
        for row in t.rows:
            for cell in row.cells:
                for p in cell.paragraphs:
                    m = _IMG_MARKER.search(p.text)
                    if m:
                        key = m.group(1)
                        _put_image_in_paragraph(p, images_map.get(key))

# ---------- formatters ----------
def _fmt2(x, nd=2, dash="—"):
    try:
        if x is None: return dash
        xf = float(x)
        if xf != xf:  # NaN
            return dash
        return f"{xf:.{nd}f}"
    except Exception:
        return dash

def _fmt_ratio_or_pct(x, pct=False, dash="—", nd=2):
    try:
        if x is None:
            return dash
        xf = float(x)
        if pct:
            return f"{xf*100:.{nd}f}%"
        return f"{xf:.{nd}f}"
    except Exception:
        return dash

def _fmt_list(lst, *, max_items=20, sep=", "):
    if not lst:
        return "—"
    if isinstance(lst, (str, bytes)):
        return str(lst)
    try:
        seq = list(lst)
    except Exception:
        return str(lst)
    if len(seq) <= max_items:
        return sep.join(map(str, seq))
    head = sep.join(map(str, seq[:max_items]))
    return f"{head}{sep}… (+{len(seq)-max_items} more)"


# put this right below _fmt_list(...)
def _fmt_list_or_message(lst, *, empty_msg="None (no issues detected)", max_items=20, sep=", "):
    if not lst:
        return empty_msg
    try:
        seq = list(lst) if not isinstance(lst, (str, bytes)) else [lst]
    except Exception:
        return str(lst)
    if len(seq) <= max_items:
        return sep.join(map(str, seq))
    head = sep.join(map(str, seq[:max_items]))
    return f"{head}{sep}… (+{len(seq)-max_items} more)"


def _fmt_feature_names(v, *, max_names=30):
    if v is None:
        return ""
    if isinstance(v, (list, tuple)):
        if len(v) <= max_names:
            return ", ".join(map(str, v))
        head = ", ".join(map(str, v[:max_names]))
        return f"{head}, … (+{len(v)-max_names} more)"
    return str(v)

def _fmt_target_balance(tb):
    if not isinstance(tb, dict) or not tb:
        return ""
    vals = list(tb.values())
    are_probs = all(isinstance(x, (int, float)) and 0 <= float(x) <= 1 for x in vals)
    items = []
    for k in sorted(tb.keys()):
        v = tb[k]
        if are_probs:
            try:
                items.append(f"{k}: {float(v)*100:.1f}%")
            except Exception:
                items.append(f"{k}: {v}")
        else:
            items.append(f"{k}: {v}")
    return ", ".join(items)

# ---------- SHAP helpers ----------
def _attach_shap_to_context(results: dict, context: dict) -> None:
    shap_ctx = {
        "shap_section": False,
        "shap_beeswarm_path": None,
        "shap_bar_path": None,
        "shap_top_features": [],
    }
    try:
        shap_res = (results or {}).get("SHAPCheck") or {}
        if "SHAPCheck" in shap_res and isinstance(shap_res["SHAPCheck"], dict):
            shap_res = shap_res["SHAPCheck"]
        plots = shap_res.get("plots") or {}
        bees = plots.get("beeswarm")
        barp = plots.get("bar")
        legacy = shap_res.get("shap_plot_path")
        if not bees and legacy:
            bees = legacy
        shap_ctx["shap_beeswarm_path"] = bees if bees and os.path.exists(bees) else None
        shap_ctx["shap_bar_path"] = barp if barp and os.path.exists(barp) else None
        shap_ctx["shap_top_features"] = shap_res.get("top_features") or []
        shap_ctx["shap_section"] = bool(
            shap_ctx["shap_beeswarm_path"] or shap_ctx["shap_bar_path"] or shap_ctx["shap_top_features"]
        )
    except Exception as e:
        print("⚠️ SHAP context attach failed:", e)
    context.update(shap_ctx)

# --- main class --------------------------------------------------------------
class ReportBuilder:
    """Build a Word report from validation results (no docxtpl)."""

    def __init__(self, results, template_path, output_path):
        self.results = results or {}
        self.template_path = template_path or files("tanml.report.templates").joinpath("report_template.docx")
        self.output_path = output_path

        corr = self.results.get("CorrelationCheck", {}) or {}
        corr_artifacts = corr.get("artifacts", corr)
        self.corr_heatmap_path = corr_artifacts.get("heatmap_path")
        self.corr_pearson_path = corr_artifacts.get("pearson_csv", "N/A")
        self.corr_spearman_path = corr_artifacts.get("spearman_csv", "N/A")

    def _grab(self, name, default=None):
        return (
            self.results.get(name)
            or self.results.get("check_results", {}).get(name)
            or default
        )

    @staticmethod
    def _extract_baseline_logit_rows(ctx):
        """
        Read Logit baseline metrics from ctx["LogitStats"]["baseline_metrics"]
        and map to a simple 2-col table.
        """
        logit = (ctx.get("LogitStats") or {})
        bm = logit.get("baseline_metrics") or {}
        summary = bm.get("summary") or bm  # support both shapes

        order = [
            ("ROC-AUC",      ("AUC","auc","roc_auc","rocauc")),
            ("KS",           ("KS","ks")),
            ("F1",           ("F1","f1","f1_score")),
            ("PR AUC",       ("PR_AUC","pr_auc","average_precision","prauc")),
            ("Gini",         ("GINI","gini")),
            ("Precision",    ("Precision","precision","prec")),
            ("Recall",       ("Recall","recall","tpr","sensitivity")),
            ("Accuracy",     ("Accuracy","accuracy","acc")),
            ("Brier Score",  ("Brier","brier","brier_score")),
            ("Log Loss",     ("Log Loss","log_loss","logloss")),
        ]
        rows = []
        for label, keys in order:
            for k in keys:
                if k in summary and summary[k] is not None:
                    v = summary[k]
                    try:
                        v = f"{float(v):.4f}"
                    except Exception:
                        v = str(v)
                    rows.append({"metric": label, "value": v})
                    break
        return rows

    def build(self):
        # ===== 1) Build context =================================================
        self.results.setdefault("RuleEngineCheck", AttrDict({"rules": {}, "overall_pass": True}))
        ctx = pycopy.deepcopy(self.results)
        ctx.update(self.results.get("check_results", {}))

        # unwrap {"Key":{"Key":...}}
        for k, v in list(ctx.items()):
            if isinstance(v, dict) and k in v and len(v) == 1:
                ctx[k] = v[k]

        # defaults
        for k, note in [
            ("RawDataCheck", "Raw-data check skipped"),
            ("CleaningReproCheck", "Cleaning-repro check skipped"),
        ]:
            ctx.setdefault(k, AttrDict({"note": note}))

        # Model meta fallback
        if "ModelMetaCheck" not in ctx or "model_type" not in ctx["ModelMetaCheck"]:
            meta_fields = [
                "model_type", "model_class", "module", "n_features", "feature_names",
                "n_train_rows", "target_balance", "hyperparam_table", "attributes",
            ]
            meta = {f: self.results.get(f) for f in meta_fields if self.results.get(f) is not None}
            ctx["ModelMetaCheck"] = AttrDict(meta or {"note": "Model metadata not available"})

        # SHAP
        _attach_shap_to_context(self.results, ctx)

        # EDA
        eda = self._grab("EDACheck", {})
        ctx["eda_summary_path"] = eda.get("summary_stats", "N/A")
        ctx["eda_missing_path"] = eda.get("missing_values", "N/A")
        ctx["eda_images_paths"] = [
            os.path.join("reports/eda", fn)
            for fn in (eda.get("visualizations", []) or [])
            if os.path.exists(os.path.join("reports/eda", fn))
        ]

        # Correlation
        corr_res = self._grab("CorrelationCheck", {}) or {}
        corr_artifacts = corr_res.get("artifacts", corr_res)
        corr_summary = corr_res.get("summary", {}) or {}

        def _pick(d, *keys, default=None):
            for k in keys:
                if k in d and d[k] is not None:
                    return d[k]
            return default

        top_pairs_csv = (
            corr_artifacts.get("top_pairs_csv")
            or corr_artifacts.get("pairs_csv")
            or corr_artifacts.get("csv")
        )
        top_pairs_main_csv = (
            corr_artifacts.get("top_pairs_main_csv")
            or corr_artifacts.get("main_csv")
        )
        heatmap_path = (
            corr_artifacts.get("heatmap_path")
            or corr_artifacts.get("heatmap")
            or self.corr_heatmap_path
        )

        method = _pick(corr_summary, "method", "corr_method")
        threshold = _pick(corr_summary, "threshold", "high_corr_threshold")
        n_numeric_features = _pick(corr_summary, "n_numeric_features", "numeric_features", "n_features_numeric")
        plotted_features = _pick(corr_summary, "plotted_features", "features_plotted")
        plotted_full_matrix = _pick(corr_summary, "plotted_full_matrix", "full_matrix")
        n_pairs_total = _pick(corr_summary, "n_pairs_total", "pairs_total")
        n_pairs_flagged = _pick(corr_summary, "n_pairs_flagged_ge_threshold", "n_pairs_flagged")

        ctx.setdefault("CorrelationCheck", {})
        ctx["CorrelationCheck"].update({
            "method": method,
            "threshold": threshold,
            "n_numeric_features": n_numeric_features,
            "plotted_features": plotted_features,
            "plotted_full_matrix": plotted_full_matrix,
            "n_pairs_total": n_pairs_total,
            "n_pairs_flagged_ge_threshold": n_pairs_flagged,
            "top_pairs_csv": top_pairs_csv,
            "top_pairs_main_csv": top_pairs_main_csv,
        })
        ctx["correlation_pearson_path"] = self.corr_pearson_path
        ctx["correlation_spearman_path"] = self.corr_spearman_path
        ctx["correlation_heatmap_path"] = heatmap_path

        corr_preview_headers = [
            "feature_i", "feature_j", "corr",
            "n_used", "pct_missing_i", "pct_missing_j",
        ]
        pairs_csv_for_preview = top_pairs_main_csv or top_pairs_csv
        corr_preview_rows = _read_csv_as_rows(pairs_csv_for_preview, max_rows=None)

        # round numeric fields — esp. 'corr' to 4 decimals
        rounded_corr_rows = []
        for r in corr_preview_rows:
            new_row = {}
            for h in corr_preview_headers:
                val = r.get(h, "")
                if h == "corr":
                    try:
                        new_row[h] = f"{float(val):.4f}"
                    except Exception:
                        new_row[h] = val
                elif h in ("pct_missing_i", "pct_missing_j"):
                    try:
                        new_row[h] = f"{float(val):.2f}"
                    except Exception:
                        new_row[h] = val
                else:
                    new_row[h] = val
            rounded_corr_rows.append(new_row)

        corr_preview_rows = rounded_corr_rows



        # ---- Fallback: if no top-pairs CSV rows, derive from pearson/spearman matrix ----
        if not corr_preview_rows:
            cc_summary = (ctx.get("CorrelationCheck") or {}).get("summary", {}) or {}
            thr = cc_summary.get("threshold")
            if thr is None:
                # Also try the normalized keys we stored earlier
                thr = (ctx.get("CorrelationCheck") or {}).get("threshold") or 0.80

            # Prefer pearson; fall back to spearman
            pearson_path = ctx.get("correlation_pearson_path")
            spearman_path = ctx.get("correlation_spearman_path")

            fallback_rows = []
            if pearson_path and os.path.exists(pearson_path):
                fallback_rows = _fallback_pairs_from_corr_matrix(pearson_path, float(thr), top_k=200)
            elif spearman_path and os.path.exists(spearman_path):
                fallback_rows = _fallback_pairs_from_corr_matrix(spearman_path, float(thr), top_k=200)

            # Use fallback if we found any
            if fallback_rows:
                corr_preview_rows = fallback_rows


        # ---------- Performance (classification) ----------
        perf_root = self.results.get("performance", {}) or {}
        cls = perf_root.get("classification", {}) or {}
        cls_summary = (cls.get("summary") or {})
        cls_plots = (cls.get("plots") or {})
        cls_tables = (cls.get("tables") or {})

        def _pick_first(d, *keys):
            for k in keys:
                if k in d and d[k] is not None:
                    return d[k]
            low = {str(k).lower(): v for k, v in d.items()}
            for k in keys:
                lk = str(k).lower()
                if lk in low and low[lk] is not None:
                    return low[lk]
            return None

        ctx["classification_summary"] = {
            "AUC":       _pick_first(cls_summary, "AUC", "auc", "roc_auc"),
            "KS":        _pick_first(cls_summary, "KS", "ks"),
            "GINI":      _pick_first(cls_summary, "GINI", "gini"),
            "PR_AUC":    _pick_first(cls_summary, "PR_AUC", "pr_auc", "average_precision"),
            "F1":        _pick_first(cls_summary, "F1", "f1", "f1_score"),
            "Precision": _pick_first(cls_summary, "Precision", "precision"),
            "Recall":    _pick_first(cls_summary, "Recall", "recall", "tpr", "sensitivity"),
            "Accuracy":  _pick_first(cls_summary, "Accuracy", "accuracy"),
            "Brier":     _pick_first(cls_summary, "Brier", "brier", "brier_score"),
        }
        ctx["classification_tables"] = {
            "confusion": _read_csv_as_rows(_pick_first(cls_tables, "confusion_csv", "confusion")),
            "lift":      _read_csv_as_rows(_pick_first(cls_tables, "lift_csv", "decile_lift_csv", "gain_lift_csv")),
        }
        ctx["classification_plot_paths"] = {
            "roc":         _pick_first(cls_plots, "roc", "roc_curve"),
            "pr":          _pick_first(cls_plots, "pr", "pr_curve", "precision_recall"),
            "lift":        _pick_first(cls_plots, "lift", "cumulative_gain", "lift_curve"),
            "calibration": _pick_first(cls_plots, "calibration", "reliability"),
            "confusion":   _pick_first(cls_plots, "confusion", "confusion_matrix"),
            "ks":          _pick_first(cls_plots, "ks", "ks_curve"),
        }

        # Regression metrics
        reg = self._grab("RegressionMetrics", {}) or {}
        reg_art = reg.get("artifacts", {}) or {}
        ctx.setdefault("RegressionMetrics", {})
        ctx["RegressionMetrics"].update({
            "notes": reg.get("notes", []),
            "rmse": reg.get("rmse"),
            "mae": reg.get("mae"),
            "median_ae": reg.get("median_ae"),
            "r2": reg.get("r2"),
            "r2_adjusted": reg.get("r2_adjusted"),
            "mape_or_smape": reg.get("mape_or_smape"),
            "mape_used": reg.get("mape_used"),
            "artifacts": {
                "pred_vs_actual": reg_art.get("pred_vs_actual"),
                "residuals_vs_pred": reg_art.get("residuals_vs_pred"),
                "residual_hist": reg_art.get("residual_hist"),
                "qq_plot": reg_art.get("qq_plot"),
                "abs_error_box": reg_art.get("abs_error_box"),
                "abs_error_violin": reg_art.get("abs_error_violin"),
            }
        })

        # Summary (ensure)
        ctx.setdefault("summary", {})
        ctx["summary"].setdefault("rmse", ctx["RegressionMetrics"].get("rmse"))
        ctx["summary"].setdefault("mae",  ctx["RegressionMetrics"].get("mae"))
        ctx["summary"].setdefault("r2",   ctx["RegressionMetrics"].get("r2"))
        if "task_type" not in ctx:
            ctx["task_type"] = "regression" if ctx["RegressionMetrics"].get("rmse") is not None else "classification"

        # Stress list→dict back-compat
        if isinstance(self.results.get("StressTestCheck"), list):
            ctx["StressTestCheck"] = {"table": self.results["StressTestCheck"]}

        # Input Cluster Coverage
        icc = self._grab("InputClusterCoverageCheck", {}) or self._grab("InputClusterCheck", {}) or {}
        icc_art = icc.get("artifacts", icc)
        cluster_rows = icc.get("cluster_table") or icc_art.get("cluster_table") or []
        cluster_csv = (icc.get("cluster_csv") or icc_art.get("cluster_csv") or icc_art.get("csv") or "—")
        plot_path = (icc.get("cluster_plot_img") or icc_art.get("cluster_plot_img") or icc_art.get("plot_img"))

        ctx.setdefault("InputClusterCheck", {})
        ctx["InputClusterCheck"]["cluster_csv"] = cluster_csv
        ctx["InputClusterCheck"]["cluster_table"] = [
            {
                "Cluster": r.get("Cluster") or r.get("cluster") or r.get("label"),
                "Count": r.get("Count") or r.get("count"),
                "Percent": r.get("Percent") or r.get("percent"),
            }
            for r in cluster_rows if isinstance(r, dict)
        ]
        ctx["ks_curve_path"] = ctx["classification_plot_paths"].get("ks")

        # VIF normalize
        vif = self._grab("VIFCheck", {})
        ctx["VIFCheck"] = AttrDict(
            vif if isinstance(vif, dict) and "vif_table" in vif
            else {"vif_table": [], "high_vif_features": [], "error": "Invalid VIFCheck"}
        )

        # ===== 2) Open template, replace text placeholders & images ==========
        doc = Document(str(self.template_path))

        # -------- scalar_map (text placeholders) ----------------------------
        scalar_map = {
            "validation_date": ctx.get("validation_date", "") or "",
            "validated_by": ctx.get("validated_by", "") or "",
            "model_path": ctx.get("model_path", "") or "",
            "task_type": (ctx.get("task_type") or "classification").title(),
            "ModelMetaCheck.model_class":  (ctx.get("ModelMetaCheck", {}) or {}).get("model_class", ""),   
            "ModelMetaCheck.module":       (ctx.get("ModelMetaCheck", {}) or {}).get("module", ""),        
            "ModelMetaCheck.model_type":   (ctx.get("ModelMetaCheck", {}) or {}).get("model_type", ""),
            "ModelMetaCheck.n_features":   (ctx.get("ModelMetaCheck", {}) or {}).get("n_features", ""),
            "ModelMetaCheck.feature_names": _fmt_feature_names((ctx.get("ModelMetaCheck", {}) or {}).get("feature_names")),
            "ModelMetaCheck.n_train_rows": (ctx.get("ModelMetaCheck", {}) or {}).get("n_train_rows", ""),
            "ModelMetaCheck.target_balance": _fmt_target_balance((ctx.get("ModelMetaCheck", {}) or {}).get("target_balance")),
        }

        # --- Logistic (Logit) summary text for classification template ---
        logit_ctx = ctx.get("LogitStats") or {}
        scalar_map["LogitStats.summary_text"] = logit_ctx.get("summary_text") or ""

        # REGRESSION rounded values / labels
        scalar_map.update({
            "summary.rmse2": _fmt2(ctx.get("summary", {}).get("rmse")),
            "summary.mae2":  _fmt2(ctx.get("summary", {}).get("mae")),
            "summary.r22":   _fmt2(ctx.get("summary", {}).get("r2")),
            "RegressionMetrics.r2_adjusted2": _fmt2(ctx.get("RegressionMetrics", {}).get("r2_adjusted")),
            "RegressionMetrics.median_ae2":   _fmt2(ctx.get("RegressionMetrics", {}).get("median_ae")),
        })

        scalar_map.update({
            "eda_summary_path": ctx.get("eda_summary_path") or "(not available)",
            "eda_missing_path": ctx.get("eda_missing_path") or "(not available)",
        })
        rm = ctx.get("RegressionMetrics", {}) or {}
        mape_or_smape = rm.get("mape_or_smape")
        mape_used = rm.get("mape_used")
        scalar_map["RegressionMetrics.mape_label"] = (
            "N/A" if mape_or_smape is None else f"{_fmt2(mape_or_smape)}% ({'MAPE' if mape_used else 'SMAPE'})"
        )
        notes_list = rm.get("notes") or []
        notes_text = "\n".join(map(str, notes_list)).strip() if notes_list else ""
        scalar_map["RegressionMetrics.notes_text"] = notes_text if notes_text else "None"

        # CLASSIFICATION rounded values
        cs = ctx.get("classification_summary", {}) or {}
        scalar_map.update({
            "classification_summary.AUC2":       _fmt2(cs.get("AUC")),
            "classification_summary.KS2":        _fmt2(cs.get("KS")),
            "classification_summary.F12":        _fmt2(cs.get("F1")),
            "classification_summary.PR_AUC2":    _fmt2(cs.get("PR_AUC")),
            "classification_summary.GINI2":      _fmt2(cs.get("GINI")),
            "classification_summary.Precision2": _fmt2(cs.get("Precision")),
            "classification_summary.Recall2":    _fmt2(cs.get("Recall")),
            "classification_summary.Accuracy2":  _fmt2(cs.get("Accuracy")),
            "classification_summary.Brier2":     _fmt2(cs.get("Brier")),
        })

        # DataQualityCheck (train/test)
        dq = ctx.get("DataQualityCheck") or {}
        def _pick_dq(dq_root, split, key, fallback_key=None):
            split_d = dq_root.get(split) or {}
            if key in split_d and split_d[key] is not None:
                return split_d[key]
            if fallback_key and (fallback_key in dq_root) and dq_root[fallback_key] is not None:
                return dq_root[fallback_key]
            alt = {
                ("avg_missing",): ["avg_missing_rate", "missing_rate"],
                ("columns_with_missing",): ["cols_with_missing", "missing_columns"],
            }
            for k_tuple, alts in alt.items():
                if key in k_tuple:
                    for a in alts:
                        if split_d.get(a) is not None:
                            return split_d[a]
                        if fallback_key and dq_root.get(a) is not None:
                            return dq_root[a]
            return None

        train_avg_missing = _pick_dq(dq, "train", "avg_missing", fallback_key="avg_missing")
        test_avg_missing  = _pick_dq(dq, "test",  "avg_missing", fallback_key="avg_missing")
        train_cols_mis = _pick_dq(dq, "train", "columns_with_missing", fallback_key="columns_with_missing")
        test_cols_mis  = _pick_dq(dq, "test",  "columns_with_missing", fallback_key="columns_with_missing")
        const_cols_dq = dq.get("constant_columns")
        if const_cols_dq is None:
            const_cols_dq = (dq.get("train") or {}).get("constant_columns") or (dq.get("test") or {}).get("constant_columns")

        scalar_map.update({
            "DataQualityCheck.train_avg_missing": _fmt_ratio_or_pct(train_avg_missing, pct=True, dash=_p("unknown")),
            "DataQualityCheck.test_avg_missing":  _fmt_ratio_or_pct(test_avg_missing,  pct=True, dash=_p("unknown")),
            "DataQualityCheck.train_cols_missing": _fmt_list_or_message(train_cols_mis, empty_msg=_p("none_detected"), max_items=25),
            "DataQualityCheck.test_cols_missing":  _fmt_list_or_message(test_cols_mis,  empty_msg=_p("none_detected"), max_items=25),
            "DataQualityCheck.constant_columns_str": _fmt_list_or_message(const_cols_dq, empty_msg=_p("none_detected"), max_items=25),
        })

        # RawDataCheck
        rd = ctx.get("RawDataCheck") or {}
        total_rows = rd.get("total_rows") or rd.get("n_rows") or rd.get("rows")
        total_cols = rd.get("total_columns") or rd.get("n_columns") or rd.get("columns")
        avg_missing = rd.get("avg_missing") or rd.get("avg_missing_rate") or rd.get("missing_rate")
        dup_rows = rd.get("duplicate_rows") or rd.get("n_duplicate_rows") or rd.get("duplicates")
        cols_with_missing = rd.get("columns_with_missing") or rd.get("cols_with_missing") or rd.get("missing_columns")
        const_cols = rd.get("constant_columns") or rd.get("constant")
        raw_skipped = bool(rd.get("note")) and all(
            v is None for v in [total_rows, total_cols, avg_missing, dup_rows, cols_with_missing, const_cols]
        )

        scalar_map.update({
            "RawDataCheck.total_rows":
                _p("not_provided") if raw_skipped else (total_rows if total_rows is not None else _p("unknown")),
            "RawDataCheck.total_columns":
                _p("not_provided") if raw_skipped else (total_cols if total_cols is not None else _p("unknown")),
            "RawDataCheck.avg_missing_pct":
                _p("not_provided") if raw_skipped else _fmt_ratio_or_pct(avg_missing, pct=True, dash=_p("unknown")),
            "RawDataCheck.columns_with_missing_str":
                _p("not_provided") if raw_skipped else _fmt_list_or_message(cols_with_missing, empty_msg=_p("none_detected"), max_items=30),
            "RawDataCheck.duplicate_rows":
                _p("not_provided") if raw_skipped else (str(dup_rows) if dup_rows not in (None, 0) else _p("none_detected")),
            "RawDataCheck.constant_columns_str":
                _p("not_provided") if raw_skipped else _fmt_list_or_message(const_cols, empty_msg=_p("none_detected"), max_items=30),
        })


        # Correlation formatted fields & notes
        cc = (ctx.get("CorrelationCheck") or {})
        def _fmt_int(x, dash="—"):
            try:
                if x is None: return dash
                return str(int(round(float(x))))
            except Exception:
                return dash
        def _fmt_float(x, nd=2, dash="—"):
            try:
                if x is None: return dash
                return f"{float(x):.{nd}f}"
            except Exception:
                return dash

        notes_val = cc.get("notes")
        if isinstance(notes_val, (list, tuple)):
            notes_text = "Notes:\n" + "\n".join(f"- {str(n)}" for n in notes_val) if notes_val else ""
        elif isinstance(notes_val, str):
            notes_text = "Notes:\n- " + notes_val if notes_val.strip() else ""
        else:
            notes_text = ""

        scalar_map.update({
            "CorrelationCheck.method2": cc.get("method", "") or "",
            "CorrelationCheck.threshold2": _fmt_float(cc.get("threshold")),
            "CorrelationCheck.plotted_features2": _fmt_int(cc.get("plotted_features")),
            "CorrelationCheck.n_numeric_features2": _fmt_int(cc.get("n_numeric_features")),
            "CorrelationCheck.n_pairs_flagged_ge_threshold2": _fmt_int(cc.get("n_pairs_flagged_ge_threshold")),
            "CorrelationCheck.n_pairs_total2": _fmt_int(cc.get("n_pairs_total")),
            "CorrelationCheck.top_pairs_csv_path": cc.get("top_pairs_csv") or cc.get("top_pairs_main_csv") or "",
            "CorrelationCheck.notes_text": notes_text,
        })
        scalar_map.update({
            "correlation_pearson_path": ctx.get("correlation_pearson_path", "") or "",
            "correlation_spearman_path": ctx.get("correlation_spearman_path", "") or "",
        })

        # VIF fallback text
        vif_ctx = ctx.get("VIFCheck") or {}
        vif_rows = vif_ctx.get("vif_table") or []
        scalar_map["VIFCheck.note_text"] = "" if (isinstance(vif_rows, list) and len(vif_rows) > 0) else "_No VIF results were generated for this run._"

        # Stress fallback text
        st_rows = (ctx.get("StressTestCheck") or {}).get("table") or []
        scalar_map["StressTest.note_text"] = "" if (isinstance(st_rows, list) and len(st_rows) > 0) else "_No stress-test results were generated for this run._"

        # InputCluster fallback text
        icc_rows = ctx.get("InputClusterCheck", {}).get("cluster_table") or []
        scalar_map["InputClusterCheck.note_text"] = "" if (isinstance(icc_rows, list) and len(icc_rows) > 0) else "_No cluster summary was generated for this run._"
        scalar_map["InputClusterCheck.cluster_csv"] = ctx.get("InputClusterCheck", {}).get("cluster_csv") or "(not available)"

        # RuleEngineCheck pretty text
        rec = ctx.get("RuleEngineCheck") or {}
        rules = rec.get("rules") or {}
        def _fmt_rule_value(v):
            if isinstance(v, bool):
                return "✅ Pass" if v else "❌ Fail"
            return str(v)
        overall_pass_val = rec.get("overall_pass")
        if isinstance(overall_pass_val, bool):
            overall_pass_str = "✅ Yes" if overall_pass_val else "❌ No"
        else:
            overall_pass_str = str(overall_pass_val) if overall_pass_val is not None else "—"
        if isinstance(rules, dict) and rules:
            lines = [f"- {str(k)}: {_fmt_rule_value(v)}" for k, v in rules.items()]
            rules_text = "\n".join(lines)
        else:
            rules_text = "No rule results were generated for this run."
        scalar_map.update({
            "RuleEngineCheck.overall_pass_str": overall_pass_str,
            "RuleEngineCheck.rules_text": rules_text,
        })

        # ModelMeta hyperparams & attributes text
        mm = ctx.get("ModelMetaCheck") or {}
        hyper_table = mm.get("hyperparam_table") or []
        if isinstance(hyper_table, list) and hyper_table:
            hp_lines = []
            for row in hyper_table:
                try:
                    k = row.get("param"); v = row.get("value")
                except Exception:
                    k, v = None, None
                if k is not None:
                    hp_lines.append(f"- {k}: {v}")
            hyperparams_text = "\n".join(hp_lines) if hp_lines else "_No hyperparameters provided._"
        else:
            hyperparams_text = "_No hyperparameters provided._"

        attrs = mm.get("attributes")
        if isinstance(attrs, dict) and attrs:
            attr_lines = [f"- {k}: {v}" for k, v in attrs.items()]
            attributes_text = "\n".join(attr_lines)
        else:
            attributes_text = (str(attrs) if attrs not in (None, "", []) else "_No attributes available._")

        scalar_map.update({
            "ModelMetaCheck.hyperparams_text": hyperparams_text,
            "ModelMetaCheck.attributes_text": attributes_text,
        })

        # Logistic stats (optional)
        lsf = ctx.get("LogisticStatsFit") or {}
        scalar_map.update({
            "LogisticStatsFit.log_lik2":   _fmt2(lsf.get("log_lik")),
            "LogisticStatsFit.aic2":       _fmt2(lsf.get("aic")),
            "LogisticStatsFit.bic2":       _fmt2(lsf.get("bic")),
            "LogisticStatsFit.pseudo_r22": _fmt2(lsf.get("pseudo_r2")),
            "LogisticStatsSummary_text": (
                ctx.get("LogisticStatsSummary_text")
                or ctx.get("LogisticStatsSummary")
                or "Logistic diagnostics not available."
            ),
        })

        # Linear (OLS) stats (optional)
        lin = ctx.get("LinearStats") or {}
        scalar_map["LinearStats.summary_text"] = lin.get("summary_text", "") or ""

        # EDA inclusion controls
        all_eda = ctx.get("eda_images_paths") or []
        opts = ctx.get("report_options") or {}
        K = opts.get("eda_max_images")
        if K is None:
            K = len(all_eda)
        else:
            try:
                K = int(K)
            except Exception:
                K = len(all_eda)
        eda_cols = int(opts.get("eda_grid_cols", 3))
        eda_subset = all_eda[:K] if K >= 0 else all_eda
        scalar_map["eda_count_note"] = f"(showing {len(eda_subset)} of {len(all_eda)})" if len(all_eda) != len(eda_subset) else ""
        ctx["_eda_subset"] = eda_subset
        ctx["_eda_cols"] = eda_cols

        # text replacements
        _replace_text_placeholders(doc, scalar_map)

        # image markers
        images_map = {
            "roc": ctx["classification_plot_paths"].get("roc"),
            "pr": ctx["classification_plot_paths"].get("pr"),
            "lift": ctx["classification_plot_paths"].get("lift"),
            "calibration": ctx["classification_plot_paths"].get("calibration"),
            "confusion": ctx["classification_plot_paths"].get("confusion"),
            "ks": ctx.get("ks_curve_path"),
            "correlation_heatmap": ctx.get("correlation_heatmap_path"),
            "shap_beeswarm": ctx.get("shap_beeswarm_path"),
            "shap_bar": ctx.get("shap_bar_path"),
            "reg_pred_vs_actual": ctx["RegressionMetrics"]["artifacts"].get("pred_vs_actual"),
            "reg_residuals_vs_pred": ctx["RegressionMetrics"]["artifacts"].get("residuals_vs_pred"),
            "reg_residual_hist": ctx["RegressionMetrics"]["artifacts"].get("residual_hist"),
            "reg_qq": ctx["RegressionMetrics"]["artifacts"].get("qq_plot"),
            "reg_abs_error_box": ctx["RegressionMetrics"]["artifacts"].get("abs_error_box"),
            "reg_abs_error_violin": ctx["RegressionMetrics"]["artifacts"].get("abs_error_violin"),
            "cluster_plot": plot_path,
            "logit_summary": ctx.get("logit_summary_path"),
        }
        _replace_image_markers(doc, images_map, width_in=4.8)

        # save once before inserting tables/grids by anchors
        doc.save(self.output_path)

        # ===== 3) Post-render insertions via anchors ==========================
        docx = Document(self.output_path)

        # EDA grid
        insert_image_grid(
            docx,
            anchor="Distribution Plots",
            img_paths=ctx.get("_eda_subset") or [],
            cols=ctx.get("_eda_cols") or 3,
            width_in=2.2
        )

        # helpers
        def _round_row_numbers(row, places=2):
            if not isinstance(row, dict):
                return row
            out = {}
            for k, v in row.items():
                try:
                    if v is None:
                        out[k] = v
                    else:
                        fv = float(v)
                        out[k] = round(fv, places) if np.isfinite(fv) else v
                except Exception:
                    out[k] = v
            return out

        stress_rows = (ctx.get("StressTestCheck", {}) or {}).get("table", []) or []
        stress_rows = [_round_row_numbers(r) for r in stress_rows]

        # OLS coefficients (rounded)
        ols_coeff_rows = (ctx.get("LinearStats") or {}).get("coeff_table", []) or []
        ols_coeff_rows = [_round_row_numbers(r, places=4) for r in ols_coeff_rows]

        # Ensure logistic minimal coef table exists/rounded (legacy 2-col)
        if "coef_table" not in ctx or not isinstance(ctx["coef_table"], list):
            ctx["coef_table"] = []
        logit_coeff_rows = ctx.get("coef_table", []) or []
        logit_coeff_rows = [_round_row_numbers(r, places=4) for r in logit_coeff_rows]
        ctx["coef_table"] = logit_coeff_rows

        # ---------- Build full logistic coefficients table (OLS-like) ----------
        def _coerce_float_or_blank(v):
            try:
                if v is None:
                    return ""
                fv = float(v)
                return fv if np.isfinite(fv) else ""
            except Exception:
                return ""

        logit_full_rows = []

        
        stats_full = None
        for source_key in ("LogisticStats", "LogisticStatsCheck"):
            cand = self._grab(source_key) or {}
            if isinstance(cand, dict) and isinstance(cand.get("coef_table_full"), list):
                stats_full = cand["coef_table_full"]
                break
        if isinstance(stats_full, list) and stats_full:
            for r in stats_full:
                logit_full_rows.append({
                    "feature":    str(r.get("feature")),
                    "coef":       _coerce_float_or_blank(r.get("coef")),
                    "std err":    _coerce_float_or_blank(r.get("std_err")),
                    "z":          _coerce_float_or_blank(r.get("z")),
                    "P>|z|":      _coerce_float_or_blank(r.get("p") or r.get("p_value") or r.get("p>|z|")),
                    "ci_low":     _coerce_float_or_blank(r.get("ci_low")),
                    "ci_high":    _coerce_float_or_blank(r.get("ci_high")),
                })

        # (2) fallback: synthesize from sklearn-like attributes (coef_ / intercept_)
        if not logit_full_rows and (ctx.get("task_type") or "").lower() == "classification":
            mm = ctx.get("ModelMetaCheck") or {}
            attrs = (mm.get("attributes") or {})
            feat_names = mm.get("feature_names") or []
            def _flatten(x):
                if x is None: return None
                arr = np.array(x)
                return arr.flatten().tolist()
            flat_coefs = _flatten(attrs.get("coef_"))
            intercept = attrs.get("intercept_")
            if isinstance(flat_coefs, list) and len(flat_coefs) == len(feat_names):
                for f, c in zip(feat_names, flat_coefs):
                    logit_full_rows.append({
                        "feature": str(f),
                        "coef": _coerce_float_or_blank(c),
                        "std err": "", "z": "", "P>|z|": "", "ci_low": "", "ci_high": ""
                    })
                if intercept is not None:
                    try:
                        b0 = float(intercept[0] if isinstance(intercept, (list, tuple, np.ndarray)) else intercept)
                    except Exception:
                        b0 = None
                    logit_full_rows.insert(0, {
                        "feature": "Intercept",
                        "coef": _coerce_float_or_blank(b0),
                        "std err": "", "z": "", "P>|z|": "", "ci_low": "", "ci_high": ""
                    })

        ctx["logit_coef_full_rows"] = logit_full_rows

        # ---------- task-aware table list ----------
        task = (ctx.get("task_type") or "classification").lower()

        if task == "regression":
            stress_headers = ["feature", "perturbation", "rmse", "r2", "delta_rmse", "delta_r2"]
            tbl_specs = [
                {
                    "anchor": "Top High-Correlation Feature Pairs (|r| ≥ Threshold)",
                    "headers": ["feature_i", "feature_j", "corr", "n_used", "pct_missing_i", "pct_missing_j"],
                    "rows": corr_preview_rows,
                },
                {
                    "anchor": "Stress Testing Results",
                    "headers": stress_headers,
                    "rows": stress_rows,
                },
                {
                    "anchor": "Cluster Summary Table",
                    "headers": ["Cluster", "Count", "Percent"],
                    "rows": ctx.get("InputClusterCheck", {}).get("cluster_table", []),
                },
                {
                    "anchor": "Variance Inflation Factor (VIF) Check",
                    "headers": ["Feature", "VIF"],
                    "rows": ctx.get("VIFCheck", {}).get("vif_table", []),
                },
                {
                    "anchor": "OLS Coefficients (Regression)",
                    "headers": ["feature", "coef", "std err", "t", "P>|t|", "ci_low", "ci_high"],
                    "rows": ols_coeff_rows,
                },
                {
                    "anchor": "Top SHAP Features",
                    "headers": ["feature", "mean_abs_shap"],
                    "rows": ctx.get("shap_top_features", []) or [],
                },
            ]
        else:
            stress_headers = ["feature", "perturbation", "accuracy", "auc", "delta_accuracy", "delta_auc"]

            labels = None
            try:
                labels = (cls.get("labels") or cls_tables.get("labels") or ctx.get("classification_labels"))
            except Exception:
                labels = None

            def _mk_confusion_headers(labs):
                if not labs or not isinstance(labs, (list, tuple)) or len(labs) == 0:
                    return ["", "Pred 0", "Pred 1"]
                return [""] + [f"Pred {str(l)}" for l in labs]

            confusion_rows = ctx.get("classification_tables", {}).get("confusion", []) or []
            confusion_headers = _mk_confusion_headers(labels)

            logit_stats = ctx.get("LogitStats") or {}
            logit_headers = logit_stats.get("coef_table_headers") or ["feature","coef","std err","z","P>|z|","ci_low","ci_high"]
            logit_rows = (
                logit_stats.get("coef_table_rows")
                or ctx.get("logit_coef_full_rows")
                or ctx.get("coef_table")
                or []
            )

            baseline_logit_rows = self._extract_baseline_logit_rows(ctx)

            tbl_specs = [
                {
                    "anchor": "Top High-Correlation Feature Pairs (|r| ≥ Threshold)",
                    "headers": ["feature_i", "feature_j", "corr", "n_used", "pct_missing_i", "pct_missing_j"],
                    "rows": corr_preview_rows,
                },
                {
                    "anchor": "Stress Testing Results",
                    "headers": stress_headers,
                    "rows": stress_rows,
                },
                {
                    "anchor": "Cluster Summary Table",
                    "headers": ["Cluster", "Count", "Percent"],
                    "rows": ctx.get("InputClusterCheck", {}).get("cluster_table", []),
                },
                {
                    "anchor": "Variance Inflation Factor (VIF) Check",
                    "headers": ["Feature", "VIF"],
                    "rows": ctx.get("VIFCheck", {}).get("vif_table", []),
                },
                {
                    "anchor": "Confusion Matrix (Classification)",
                    "headers": confusion_headers,
                    "rows": confusion_rows if confusion_rows else [{"": "(no confusion matrix available)"}],
                },
                {
                    "anchor": "Decile Lift Table",
                    "headers": ["decile", "total", "events", "avg_score", "event_rate", "lift",
                                "cum_events", "cum_total", "cum_capture_rate", "cum_population", "cum_gain"],
                    "rows": ctx.get("classification_tables", {}).get("lift", []),
                },
                {
                    "anchor": "Baseline Metrics",
                    "headers": ["metric", "value"],
                    "rows": baseline_logit_rows,
                },
                {
                    "anchor": "Logistic Regression Coefficients ",
                    "headers": logit_headers,
                    "rows": logit_rows if logit_rows else [
                        {"feature": "(no coefficients available)", "coef": "", "std err": "", "z": "", "P>|z|": "", "ci_low": "", "ci_high": ""}
                    ],
                },
                {
                    "anchor": "Feature Importances (Tree-Based Models)",
                    "headers": ["feature", "importance"],
                    "rows": ctx.get("feature_importance_table", []) or [],
                },
                {
                    "anchor": "Top SHAP Features",
                    "headers": ["feature", "mean_abs_shap"],
                    "rows": ctx.get("shap_top_features", []) or [],
                },
            ]

        for spec in tbl_specs:
            if spec["rows"]:
                if spec["anchor"] == "Decile Lift Table":
                    tbl = build_decile_lift_table(docx, spec["headers"], spec["rows"])
                else:
                    tbl = build_table(docx, spec["headers"], spec["rows"])
                insert_after(docx, spec["anchor"], tbl)
                print(f"✅ added table after «{spec['anchor']}»")


        docx.save(self.output_path)
