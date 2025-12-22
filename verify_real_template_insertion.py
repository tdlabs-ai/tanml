from docx import Document
from tanml.report.report_builder import _replace_placeholder_with_kv_table
import os
import shutil

# Copy template to avoid modifying original
template_path = "tanml/report/templates/report_template_reg.docx"
test_path = "test_real_template_output.docx"
shutil.copy(template_path, test_path)

doc = Document(test_path)
data = {"Mean": "2.40", "Std": "1.00", "Range": "-10 to 10"}

print(f"Testing replacement on COPY of real template: {test_path}")

success = _replace_placeholder_with_kv_table(doc, "{{ModelMetaCheck.target_balance}}", data)

if success:
    print("SUCCESS: Placeholder found and replaced in real template!")
else:
    print("FAILURE: Placeholder NOT found in real template.")
    
    # Debug info
    print("dumping paragraphs to check content:")
    import re
    for i, p in enumerate(doc.paragraphs):
        if "ModelMetaCheck" in p.text:
            print(f"Para {i}: {p.text}")

doc.save(test_path)
