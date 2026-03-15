# TanML: Automated Model Validation Toolkit for Tabular Machine Learning Models

[![PyPI](https://img.shields.io/pypi/v/tanml.svg)](https://pypi.org/project/tanml/)
[![Downloads](https://pepy.tech/badge/tanml)](https://pepy.tech/project/tanml)
[![Python Version](https://img.shields.io/badge/3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue?logo=python&logoColor=white)](https://pypi.org/project/tanml/)
[![OS](https://img.shields.io/badge/OS-Linux%20%7C%20macOS%20%7C%20Windows-blue?logo=linux&logoColor=white)](https://pypi.org/project/tanml/)
[![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://tdlabs-ai.github.io/tanml/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Give Feedback](https://img.shields.io/badge/Feedback-Give%20Now-blue?style=for-the-badge)](https://forms.gle/oG2JHvt7tLXE5Atu7)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17317165.svg)](https://doi.org/10.5281/zenodo.17317165)


> **TanML bridges the gap between data science tools (for building models) and governance requirements (for validating them). It's not just an ML library—it's a validation workflow with built-in documentation.**

---

### 🚀 See TanML in Action (2-Minute Walkthrough)
<video src="https://tdlabs-ai.github.io/tanml/assets/tanml_demo.mp4" width="100%" autoplay loop muted playsinline></video>

---

## 📖 Documentation

For detailed guides, API reference, and tutorials, visit our official documentation:

👉 **[https://tdlabs-ai.github.io/tanml/](https://tdlabs-ai.github.io/tanml/)**

---

## Table of Contents

- [Why TanML?](#why-tanml)
- [Install](#install)
- [Remote Access (SSH)](#remote-access-ssh)
- [Quick Start (UI)](#quick-start-ui)
- [Library Usage (Python)](#library-usage-python)
- [Testing](#testing)
- [Optional CLI Flags](#optional-cli-flags)
- [Reports](#reports)
- [Data Privacy](#data-privacy)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License & Citation](#license--citation)

---

## Why TanML?

* **End-to-end workflow:** Data Profiling → Data Preprocessing → Feature Power Ranking → Model Development → Evaluation → Reports—all in one UI.
* **Audit-ready reports:** Generate editable .docx documents for stakeholders and compliance reviews.
* **Built for regulated industries:** Designed for MRM, credit risk, insurance, and SR 11-7 contexts.
* **No code required:** Fully UI-driven—no Python knowledge needed.
* **Robust evaluation:** Drift detection, stress testing, SHAP explainability, cluster coverage.
* **Works with your stack:** Scikit-learn, XGBoost, LightGBM, CatBoost, Statsmodels.

### Comparison to Similar Tools

| Feature | TanML | Evidently AI | Deepchecks | AutoML (PyCaret/AutoGluon) | YData Profiling |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Primary Focus** | **Model Risk Management** | Monitoring | Testing / CI | Model Training | Data EDA Only |
| **Scope** | **Data + Model + Governance** | Data + Model | Data + Model | Model Building | Data Only |
| **Output** | **Audit-Ready .docx** | Dashboards | Reports | Models | HTML Report |
| **Drift Logic** | **PSI & KS (Regulatory)** | Stat. Tests | Multiple | N/A | Warnings |

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

# Upgrade an existing installation
python -m pip install --upgrade tanml
```

## Run TanML Locally

If you are running TanML directly on your own machine, start the UI with:

```bash
tanml ui
```

If the tanml command is not available in your shell, you can also try:
```
python -m tanml ui
```

Then open the following in your browser:
http://localhost:8501

## Remote Access (SSH)

If you are using standard SSH from a terminal, you may need to forward port 8501 manually. If you are using VS Code Remote SSH, VS Code may detect and forward the TanML UI port automatically.

### Standard SSH from a terminal

From your **local machine**, connect to the remote server with port forwarding enabled:

```bash
ssh -L 8501:localhost:8501 user@remote-server-ip
```

Then, on the **remote server**, run:

```bash
tanml ui
```

After that, open the following in your **local browser**:

```text
http://localhost:8501
```

### VS Code Remote SSH

If you are connected through **VS Code Remote SSH**, run `tanml ui` in the remote terminal. VS Code may automatically detect port `8501` and offer to forward it. If it does not, open the **Ports** panel in VS Code, choose **Forward a Port**, enter `8501`, and then open the forwarded local address in your browser.



---

## Quick Start (UI)

```bash
tanml ui
```

* Opens at **[http://127.0.0.1:8501](http://127.0.0.1:8501)**
* **Upload limit ~2 GB** (preconfigured)
* **Telemetry disabled by default**

## Library Usage (Python)

You can also use TanML's analysis engine directly in your Python scripts:

```python
import pandas as pd
from tanml.analysis import analyze_drift

# Load your data
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# Run drift analysis (PSI, KS Test, etc.)
results = analyze_drift(train_df, test_df)
print(results)
```

---

## Testing

TanML uses `pytest` for automated unit testing. To run tests locally:

```bash
# 1. Navigate to the project root
cd ./tanml/

# 2. Install with dev dependencies
pip install -e ".[dev]"

# 3. Run all tests
pytest tests/
```

We also use `ruff` for linting and `mypy` for type checking. These are integrated into our CI pipeline.

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

TanML generates audit ready reports (.docx) programmatically:

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

We welcome issues and PRs! See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed setup instructions.

1. **Clone & Setup**: `git clone https://github.com/tdlabs-ai/tanml.git`
2. **Environment**: `python -m venv .venv && source .venv/bin/activate`
3. **Install**: `pip install -e .[dev]`
4. **Test**: `pytest`

---

## License & Citation

**License:** MIT. See [LICENSE](https://github.com/tdlabs-ai/tanml/blob/main/LICENSE).  
SPDX-License-Identifier: MIT

© 2026 Tanmay Sah and Dolly Sah. You may use, modify, and distribute this software with appropriate attribution.

### How to cite

If TanML helps your work or publications, please cite:

> Sah, T., & Sah, D. (2026). *TanML: Automated Model Validation Toolkit for Tabular Machine Learning Models* [Software]. Zenodo. [https://doi.org/10.5281/zenodo.17317165](https://doi.org/10.5281/zenodo.17317165)

Or in BibTeX (version-agnostic):

```bibtex
@software{tanml_2026,
  author       = {Sah, Tanmay and Sah, Dolly},
  title        = {TanML: Automated Model Validation Toolkit for Tabular Machine Learning Models},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.17317165},
  url          = {https://doi.org/10.5281/zenodo.17317165},
  license      = {MIT}
}
```

A machine-readable citation file (`CITATION.cff`) is included for citation tools and GitHub’s “Cite this repository” button.

---

## AI Disclosure

Portions of this codebase (including tests, documentation, and the data_loader module) were refactored and generated with the assistance of Large Language Models (LLMs). All AI-generated contributions have been reviewed and verified by the human maintainers.

