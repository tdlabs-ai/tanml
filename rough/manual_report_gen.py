import sys
import os
import shutil
from tanml.report.report_builder import ReportBuilder

# Mock context
ctx = {
    "task_type": "regression",
    "check_results": {
        "ModelMetaCheck": {
            "model_type": "LinearRegression",
            "target_balance": {
                "Range": "-400 to 400",
                "Mean": "2.5",
                "Std": "150.0",
                "Note": "Continuous target (min/max/mean/std)"
            }
        }
    },
    "ModelMetaCheck": {}, # Top level might be present or not, but check_results is key
    "LogitStats": {},
    "LinearStats": {},
    "RegressionMetrics": {"artifacts": {}},
    "classification_plot_paths": {},
    "report_options": {},
    "feature_importance_table": [],
    "shap_top_features": [],
    "classification_tables": {},
    "eda_images_paths": [],
}

# Paths
template_path = "tanml/report/templates/report_template_reg.docx"
output_path = "manual_report_output.docx"

print("Starting manual report generation...")
try:
    rb = ReportBuilder(ctx, template_path, output_path)
    rb.build()
    print("Report built successfully.")
except Exception as e:
    print(f"Error building report: {e}")
    import traceback
    traceback.print_exc()

# Check log
log_path = "/tmp/tanml_report_debug.log"
if os.path.exists(log_path):
    print(f"\n--- LOG CONTENT ({log_path}) ---")
    with open(log_path, "r") as f:
        print(f.read())
    print("--- END LOG ---")
else:
    print(f"Log file {log_path} not found!")
