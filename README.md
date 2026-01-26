# TanML: Automated Model Validation Toolkit for Tabular Machine Learning





[![PyPI](https://img.shields.io/pypi/v/tanml.svg)](https://pypi.org/project/tanml/)
[![Downloads](https://pepy.tech/badge/tanml)](https://pepy.tech/project/tanml)
[![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://tdlabs-ai.github.io/tanml/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Give Feedback](https://img.shields.io/badge/Feedback-Give%20Now-blue?style=for-the-badge)](https://forms.gle/oG2JHvt7tLXE5Atu7)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17317165.svg)](https://doi.org/10.5281/zenodo.17317165)


> **TanML bridges the gap between data science tools (for building models) and governance requirements (for validating them). It's not just an ML library—it's a validation workflow with built-in documentation.**

* **Status:** Beta (`0.x`)
* **License:** MIT
* **Python:** 3.10–3.13
* **OS:** Linux / macOS / Windows (incl. WSL)

---

## Table of Contents

- Why TanML?
- Install
- Quick Start (UI)
- Optional CLI Flags
- Reports
- Data Privacy
- Troubleshooting
- Contributing
- License & Citation

---

## Why TanML?

* **End-to-end workflow:** Data Profiling → Preprocessing → Feature Ranking → Model Development → Evaluation → Reports—all in one UI.
* **Audit-ready Word reports:** Generate editable .docx documents for stakeholders and compliance reviews.
* **Built for regulated industries:** Designed for MRM, credit risk, insurance, and SR 11-7 contexts.
* **No code required:** Fully UI-driven—no Python knowledge needed.
* **Robust evaluation:** Drift detection, stress testing, SHAP explainability, cluster coverage.
* **Works with your stack:** scikit-learn, XGBoost, LightGBM, CatBoost.

---

## Install

We strongly recommend using a virtual environment to isolate dependencies:

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install TanML
pip install tanml
```

## Quick Start (UI)

```bash
tanml ui
```

* Opens at **[http://127.0.0.1:8501](http://127.0.0.1:8501)**
* **Upload limit ~2 GB** (preconfigured)
* **Telemetry disabled by default**

---

## Optional CLI Flags

Most users just run `tanml ui`. These help on teams/servers:

```bash
# Share on LAN
tanml ui --public

> [!WARNING]
> Using `--public` makes the UI accessible to anyone on your network. Only use this on trusted networks (like a VPN or secure office LAN) and never on public Wi-Fi.

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

**Defaults:** address `127.0.0.1`, port `8501`, limit `2048 MB`, telemetry **OFF**.

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
- **Security Note:** Running with the `--public` flag exposes the app to your local network. Ensure you are on a secure connection.
- UI artifacts and reports are written under `./tanml_runs/<session>/` in your working directory.

---

## Troubleshooting

* **Page didn’t open?** Visit `http://127.0.0.1:8501` or run `tanml ui --port 9000`.
* **Large CSVs are slow/heavy?** Prefer **Parquet**; CSV → DataFrame can use several GB RAM.
* **Artifacts missing?** Check `./tanml_runs/<session>/artifacts/`.
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






