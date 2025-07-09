# tanml/report/report_builder.py
from docxtpl import DocxTemplate, InlineImage
from docx.shared import Inches, Mm
from docx import Document
from pathlib import Path
import os, imgkit, copy as pycopy
from importlib.resources import files

TMP_DIR = Path(__file__).resolve().parents[1] / "tmp_report_assets"
TMP_DIR.mkdir(exist_ok=True)


class AttrDict(dict):
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(item)


class ReportBuilder:
    """Build a Word report from validation results."""

    def __init__(self, results, template_path, output_path):
        self.results = results
        self.template_path = template_path or files("tanml.report.templates").joinpath("report_template.docx")

        self.output_path = output_path

        corr = results.get("CorrelationCheck", {})
        self.corr_heatmap_path = corr.get("heatmap_path")
        self.corr_pearson_path = corr.get("pearson_csv", "N/A")
        self.corr_spearman_path = corr.get("spearman_csv", "N/A")

    def _grab(self, name, default=None):
        return (
            self.results.get(name)
            or self.results.get("check_results", {}).get(name)
            or default
        )

    def build(self):
        doc = DocxTemplate(str(self.template_path))


        # Ensure RuleEngineCheck exists so template never crashes
        self.results.setdefault(
            "RuleEngineCheck", AttrDict({"rules": {}, "overall_pass": True})
        )

        # Jinja context
        ctx = pycopy.deepcopy(self.results)
        ctx.update(self.results.get("check_results", {}))  # 1st-level flatten

        for k, v in list(ctx.items()):
            if isinstance(v, dict) and k in v and len(v) == 1:
                ctx[k] = v[k]

        for k, note in [
            ("RawDataCheck", "Raw-data check skipped"),
            ("CleaningReproCheck", "Cleaning-repro check skipped"),
        ]:
            ctx.setdefault(k, AttrDict({"note": note}))

        if "ModelMetaCheck" not in ctx or "model_type" not in ctx["ModelMetaCheck"]:
            meta_fields = [
                "model_type",
                "model_class",
                "module",
                "n_features",
                "feature_names",
                "n_train_rows",
                "target_balance",
                "hyperparam_table",
                "attributes",
            ]
            meta = {f: self.results.get(f) for f in meta_fields if self.results.get(f) is not None}
            ctx["ModelMetaCheck"] = AttrDict(meta or {"note": "Model metadata not available"})

        # SHAP image
        shap_path = self._grab("SHAPCheck", {}).get("shap_plot_path")
        ctx["shap_plot"] = (
            InlineImage(doc, shap_path, width=Inches(5))
            if shap_path and os.path.exists(shap_path)
            else "SHAP plot not available"
        )

        #  EDA
        eda = self._grab("EDACheck", {})
        ctx["eda_summary_path"] = eda.get("summary_stats", "N/A")
        ctx["eda_missing_path"] = eda.get("missing_values", "N/A")
        ctx["eda_images"] = [
            InlineImage(doc, os.path.join("reports/eda", fn), width=Inches(4.5))
            if os.path.exists(os.path.join("reports/eda", fn))
            else f"Missing: {fn}"
            for fn in eda.get("visualizations", [])
        ]

        # Correlation visuals
        if self.corr_heatmap_path and os.path.exists(self.corr_heatmap_path):
            ctx["correlation_heatmap"] = InlineImage(
                doc, self.corr_heatmap_path, width=Inches(5)
            )
        else:
            ctx["correlation_heatmap"] = "Heatmap not available"
        ctx["correlation_pearson_path"] = self.corr_pearson_path
        ctx["correlation_spearman_path"] = self.corr_spearman_path

        # Performance
        perf = self._grab("PerformanceCheck", {})
        if not perf:
            perf = {
                "accuracy": "N/A",
                "auc": "N/A",
                "ks": "N/A",
                "f1": "N/A",
                "confusion_matrix": [],
            }
        ctx.setdefault("check_results", {})["PerformanceCheck"] = perf
        ctx["PerformanceCheck"] = perf

        # Logistic summary image
        if "LogisticStatsCheck_obj" in self.results:
            try:
                add_logit_summary_image(
                    doc, self.results["LogisticStatsCheck_obj"], ctx, "LogitSummaryImg"
                )
            except Exception as e:
                print("⚠️  logistic summary image failed:", e)

        # VIF
        vif = self._grab("VIFCheck", {})
        ctx["VIFCheck"] = AttrDict(
            vif
            if isinstance(vif, dict) and "vif_table" in vif
            else {"vif_table": [], "high_vif_features": [], "error": "Invalid VIFCheck"}
        )

        # Stress / cluster
        if isinstance(self.results.get("StressTestCheck"), list):
            ctx["StressTestCheck"] = {"table": self.results["StressTestCheck"]}

        cluster_rows = self._grab("InputClusterCheck", {}).get("cluster_table", [])
        ctx.setdefault("InputClusterCheck", {})["cluster_table"] = [
            {
                "Cluster": r.get("Cluster") or r.get("cluster"),
                "Count": r.get("Count") or r.get("count"),
                "Percent": r.get("Percent") or r.get("percent"),
            }
            for r in cluster_rows
            if isinstance(r, dict)
        ]
        plot_path = self._grab("InputClusterCheck", {}).get("cluster_plot_img")
        ctx["InputClusterCheck"]["cluster_plot_img"] = (
            InlineImage(doc, plot_path, width=Inches(5))
            if plot_path and os.path.exists(plot_path)
            else "Plot not available"
        )

        # Render DOCX template
        print("🟢 ctx top-level keys:", list(ctx.keys()))
        print("🔍 RawDataCheck value:", ctx.get("RawDataCheck"))

        doc.render(ctx)
        doc.save(self.output_path)

        # Auto-insert tables after anchors
        tbl_specs = [
            {
                "anchor": "Stress Testing Results",
                "headers": [
                    "feature",
                    "perturbation",
                    "accuracy",
                    "auc",
                    "delta_accuracy",
                    "delta_auc",
                ],
                "rows": ctx.get("StressTestCheck", {}).get("table", []),
            },
            {
                "anchor": "Cluster Summary Table:",
                "headers": ["Cluster", "Count", "Percent"],
                "rows": ctx.get("InputClusterCheck", {}).get("cluster_table", []),
            },
            {
                "anchor": "Variance Inflation Factor (VIF) Check",
                "headers": ["Feature", "VIF"],
                "rows": ctx.get("VIFCheck", {}).get("vif_table", []),
            },
        ]

        docx = Document(self.output_path)
        for spec in tbl_specs:
            if spec["rows"]:
                tbl = build_table(docx, spec["headers"], spec["rows"])
                insert_after(docx, spec["anchor"], tbl)
                print(f"✅ added table after «{spec['anchor']}»")
        docx.save(self.output_path)


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


def insert_after(doc, anchor, tbl):
    for p in doc.paragraphs:
        if anchor.lower() in p.text.lower():
            parent = p._p.getparent()
            parent.insert(parent.index(p._p) + 1, tbl._tbl)
            return
    print(f"⚠️ anchor «{anchor}» not found")


def add_logit_summary_image(tpl_doc, sm_results, ctx, key):
    html = TMP_DIR / "logit_summary.html"
    html.write_text(sm_results.summary().as_html(), encoding="utf8")
    png = TMP_DIR / "logit_summary.png"
    imgkit.from_file(str(html), str(png), options={"quiet": ""})
    ctx[key] = InlineImage(tpl_doc, str(png), width=Mm(160))
