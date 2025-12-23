# TanML: Automated Model Validation Toolkit for Tabular Machine Learning





[![PyPI](https://img.shields.io/pypi/v/tanml.svg)](https://pypi.org/project/tanml/)
[![Downloads](https://pepy.tech/badge/tanml)](https://pepy.tech/project/tanml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Give Feedback](https://img.shields.io/badge/Feedback-Give%20Now-blue?style=for-the-badge)](https://forms.gle/oG2JHvt7tLXE5Atu7)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17317165.svg)](https://doi.org/10.5281/zenodo.17317165)

[![Watch the TanML Demo](https://img.youtube.com/vi/KlqJrVgj670/maxresdefault.jpg)](https://youtu.be/KlqJrVgj670)  
🎥 **[Watch the 5-Minute TanML Walkthrough on YouTube →](https://youtu.be/KlqJrVgj670)**  
*(End-to-end demo of UI, validation checks, and automated report generation)*


**TanML** validates tabular ML models with a zero-config **Streamlit UI** and exports an audit-ready, **editable Word report (.docx)**. It covers data quality, correlation/VIF, performance, explainability (SHAP), and robustness/stress tests—built for regulated settings (MRM, credit risk, insurance, etc.).

* **Status:** Beta (`0.x`)
* **License:** MIT
* **Python:** 3.8–3.12
* **OS:** Linux / macOS / Windows (incl. WSL)

---

## Table of Contents

- Why TanML?
- Install
- Quick Start (UI)
- What TanML Checks
- Optional CLI Flags
- Templates
- Troubleshooting
- Data Privacy
- Contributing
- License & Citation

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

## Optional CLI Flags

Most users just run `tanml ui`. These help on teams/servers:

```bash
# Share on LAN
tanml ui --public

# Different port
tanml ui --port 9000

# Headless (server/CI; no auto-open browser)
tanml ui --headless

# Larger limit (e.g., 2 GB)
tanml ui --max-mb 2048
```

Env var equivalents (Linux/macOS bash):

```bash
TANML_SERVER_ADDRESS=0.0.0.0 TANML_PORT=9000 TANML_MAX_MB=2048 tanml ui
```

Windows PowerShell:

```powershell
$env:TANML_SERVER_ADDRESS="0.0.0.0"; $env:TANML_PORT="9000"; $env:TANML_MAX_MB="2048"; tanml ui
```

**Defaults:** address `127.0.0.1`, port `8501`, limit `1024 MB`, telemetry **OFF**.

---

## Reports

TanML generates audit-ready Word reports (.docx) programmatically:

* **Model Development Report** — Cross-validation metrics, diagnostics, and performance summary
* **Model Evaluation Report** — Train/Test comparison, drift analysis, stress testing, SHAP explainability
* **Feature Power Ranking Report** — Feature importance scores, correlation analysis

Reports are generated via `tanml/ui/reports/generators.py` and exported directly from the UI.

---

## Data Privacy

- TanML runs locally; no data is sent to external services.
- Telemetry is disabled by default (and can be forced off via `--no-telemetry`).
- UI artifacts and reports are written under `./.ui_runs/<session>/` in your working directory.

---

## Troubleshooting

* **Page didn’t open?** Visit `http://127.0.0.1:8501` or run `tanml ui --port 9000`.
* **Large CSVs are slow/heavy?** Prefer **Parquet**; CSV → DataFrame can use several GB RAM.
* **Artifacts missing?** Check `./.ui_runs/<session>/artifacts/`.
* **Corporate networks:** use `tanml ui --public` to share on LAN.

---

## Contributing

We welcome issues and PRs!

- Create a virtual environment and install dev extras:
  - `python -m venv .venv && source .venv/bin/activate` (or `\.venv\Scripts\activate` on Windows)
  - `pip install -e .[dev]`
- Format/lint: `black . && isort .`
- Run tests: `pytest`

Before opening a PR, please describe the change and include a brief test or reproduction steps where applicable.

---

## License & Citation

**License:** MIT. See [LICENSE](https://github.com/tdlabs-ai/tanml/blob/main/LICENSE).  
SPDX-License-Identifier: MIT

© 2025 Tanmay Sah and Dolly Sah. You may use, modify, and distribute this software with appropriate attribution.

### How to cite

If TanML helps your work or publications, please cite:

> Sah, T., & Sah, D. (2025). *TanML: Automated Model Validation Toolkit for Tabular Machine Learning* [Software]. Zenodo. [https://doi.org/10.5281/zenodo.17317165](https://doi.org/10.5281/zenodo.17317165)

Or in BibTeX (version-agnostic):

```bibtex
@software{tanml_2025,
  author       = {Sah, Tanmay and Sah, Dolly},
  title        = {TanML: Automated Model Validation Toolkit for Tabular Machine Learning},
  year         = {2025},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.17317165},
  url          = {https://doi.org/10.5281/zenodo.17317165},
  license      = {MIT}
}
```

A machine-readable citation file (`CITATION.cff`) is included for citation tools and GitHub’s “Cite this repository” button.






