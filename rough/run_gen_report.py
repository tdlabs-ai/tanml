
import os
from tanml.report.report_builder import ReportBuilder
from docx import Document

# 1. Setup Mock Results with rich CV stats
results = {
    "summary": {
        "cv_stats": {
            "roc_auc": {
                "mean": 0.8888, "std": 0.0111, 
                "p05": 0.8700, "p50": 0.8900, "p95": 0.9000
            },
            "f1": {
                "mean": 0.7777, "std": 0.0222,
                "p05": 0.7500, "p50": 0.7800, "p95": 0.8000
            },
            "threshold_info": {
                "rule": "Max F1",
                "value": 0.5555
            }
        },
        "classification_summary": {
            "AUC": 0.88, "F1": 0.78 # simple fallback
        }
    },
    "ModelMetaCheck": {
        "model_class": "TestModel",
        "model_type": "classification"
    }
}

output_path = "test_cv_report.docx"
template_path = "tanml/report/templates/report_template_cls.docx"

print(f"Generating report from {template_path}...")

# 2. Build Report
try:
    builder = ReportBuilder(results, template_path=template_path, output_path=output_path)
    builder.build()
    print("Report built successfully.")
except Exception as e:
    print(f"FAILED to build report: {e}")
    exit(1)

# 3. Inspect the Output Doc (looking for values, verifying placeholders are gone if they existed)
# Note: The template might NOT have the {{cv.auc.mean}} placeholders yet unless the user updated the .docx file externally.
# However, if I can't check the docx content for replaced values because the template doesn't have the tags, 
# then this test only proves the code runs, not that it replaces anything.

# BUT, the user's request implied the template *would* have these sections (or they are adding them).
# I cannot verify REPLACEMENT unless the template has the tags. 
# I can verifying the MAPPING logic (done in unit test).

# Let's inspect the `scalar_map` directly again by subclassing, just to be double sure in "integration" style.
# Or better, I can verify that specific keys are NOT present in the final doc text implies... nothing if they weren't there to start.

# Assumption: I cannot modify the binary .docx template to add test tags easily without corrupting it.
# So I will rely on the unit test I already ran.
# But I will double check if the template has these tags?
# I read the template earlier (Step 107) and it DID NOT have {{cv.auc.mean}}. 
# It had `AUC-ROC: {{classification_summary.AUC2}}`.

# So running this against the CURRENT template won't show the CV stats appearing, because the template doesn't ask for them yet.
# User Logic: "I (User) have the template or will update it. You (Agent) update the code."
# Verification: "I have verified the code produces the keys."

# I will run this script just to ensure no crashes occur with the new logic present.
