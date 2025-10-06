# TanML: Automated Model Validation Toolkit for Tabular Machine Learning

[![Cite this repo](https://img.shields.io/badge/Cite-this_repo-blue)](https://github.com/tdlabs-ai/tanml#license--citation)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Downloads](https://pepy.tech/badge/tanml)](https://pepy.tech/project/tanml)

**TanML** validates tabular ML models with a zero-config **Streamlit UI** and exports an audit-ready, **editable Word report (.docx)**. It covers data quality, correlation/VIF, performance, explainability (SHAP), and robustness/stress tests—built for regulated settings (MRM, credit risk, insurance, etc.).

* **Status:** Beta (`0.x`)
* **License:** MIT
* **Python:** 3.8–3.12
* **OS:** Linux / macOS / Windows (incl. WSL)

---

## Why TanML?

* **Zero-config UI:** launch Streamlit, upload data, click **Run**—no YAML needed.
* **Audit-ready outputs:** tables/plots + a polished DOCX your stakeholders can edit.
* **Regulatory alignment:** supports common Model Risk Management themes (e.g., SR 11-7 style).
* **Works with your stack:** scikit-learn, XGBoost/LightGBM/CatBoost, etc.

---

## Install

```bash
pip install tanml
```

## Quick Start (UI)

```bash
tanml ui
```

* Opens at **[http://127.0.0.1:8501](http://127.0.0.1:8501)**
* **Upload limit ~1 GB** (preconfigured)
* **Telemetry disabled by default**

### In the app

1. **Load data** — upload a cleaned CSV/XLSX/Parquet (optional: raw or separate Train/Test).
2. **Select target & features** — target auto-suggested; features default to all non-target columns.
3. **Pick a model** — choose library/algorithm (scikit-learn, XGBoost, LightGBM, CatBoost) and tweak params.
4. **Run validation** — click **▶️ Refit & validate**.
5. **Export** — click **⬇️ Download report** to get a **DOCX** (auto-selects classification/regression template).

**Outputs**

* Report: `./.ui_runs/<session>/tanml_report_*.docx`
* Artifacts (CSV/PNGs): `./.ui_runs/<session>/artifacts/*`

---

## What TanML Checks

* **Raw Data (optional):** rows/cols, missingness, duplicates, constant columns
* **Data Quality & EDA:** summaries, distributions
* **Correlation & Multicollinearity:** heatmap, top-pairs CSV, **VIF** table
* **Performance**

  * **Classification:** AUC, PR-AUC, KS, decile lift, confusion
  * **Regression:** R², MAE, MSE/RMSE, error stats
* **Explainability:** SHAP (auto explainer; configurable background size)
* **Robustness/Stress Tests:** feature perturbations → delta-metrics
* **Model Metadata:** model class, hyperparameters, features, training info

---


## Templates

TanML ships DOCX templates (packaged in wheel & sdist):

* `tanml/report/templates/report_template_cls.docx`
* `tanml/report/templates/report_template_reg.docx`

---


## License & Citation

**License:** MIT. See [LICENSE](https://github.com/tdlabs-ai/tanml/blob/main/LICENSE).  
SPDX-License-Identifier: MIT

© 2025 Tanmay Sah and Dolly Sah. You may use, modify, and distribute this software with appropriate attribution.

### How to cite

If TanML helps your work or publications, please cite:

> Sah, T., & Sah, D. (2025). *TanML: Automated Model Validation Toolkit for Tabular Machine Learning* [Software]. Available at https://github.com/tdlabs-ai/tanml

Or in BibTeX (version-agnostic):

```bibtex
@misc{tanml,
  author = {Sah, Tanmay and Sah, Dolly},
  title  = {TanML: Automated Model Validation Toolkit for Tabular Machine Learning},
  year   = {2025},
  note   = {Software; MIT License},
  url    = {https://github.com/tdlabs-ai/tanml}
}
```

A machine-readable citation file (`CITATION.cff`) is included for citation tools and GitHub’s “Cite this repository” button.
