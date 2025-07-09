# TanML: Automated Model Validation Toolkit for Tabular Machine Learning  
[![Cite this repo](https://img.shields.io/badge/Cite-this_repo-blue)](https://github.com/tdlabs-ai/tanml)

TanML is a modular, automated model validation toolkit for tabular machine learning workflows. It supports end-to-end validation with just a YAML configuration and a single command, performing checks across data quality, robustness, explainability, and model performance.

TanML generates structured Word (DOCX) reports suitable for internal model reviews, audit documentation, and stakeholder presentations. It is designed for general-purpose ML validation and works well in domains where interpretability, fairness, and reliability are critical.

While TanML currently operates as a command-line toolkit, its architecture is designed to evolve into an intelligent validation agent—capable of integrating with AutoML pipelines, CI/CD workflows, and human-in-the-loop validation systems.

## Key Features

* One-command validation using CLI and YAML
* Supports models developed in Python (scikit-learn, XGBoost), R, or SAS
* Scenario-based flexibility:

  * Scenario A: Single model and dataset
  * Scenario B: One model per segment with separate datasets
  * Scenario C: One model and dataset with an internal segmentation column
* Comprehensive validation checks:

  * Model performance (e.g., accuracy, AUC, KS)
  * Data quality diagnostics
  * Stress testing for input robustness
  * SHAP-based explainability
  * Segment-wise validation
  * Logistic regression coefficient summaries (if applicable)
  * VIF, correlation, and multicollinearity checks
  * EDA summaries
  * Rule-based threshold validation using YAML configuration
* Professional report generation in Word (DOCX) format
* Easily extensible architecture to add custom checks or outputs

## Installation

TanML can be installed directly from PyPI using pip:

```bash
pip install tanml
```

To upgrade to the latest version:

```bash
pip install --upgrade tanml
```

After installation, you can verify the CLI is working by running:

```bash
tanml --help
```

This will display the list of available commands and options.

TanML supports Python 3.8 and above. It is recommended to use a virtual environment for clean dependency management:

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

## Folder Structure

A typical TanML project is organized as follows:

```
TanML/
├── models/                # Trained model files (e.g., .pkl, .sas7bdat)
├── data/
│   ├── raw/               # Raw data files (optional)
│   └── cleaned/           # Cleaned datasets for validation
├── examples/
│   ├── scenario_a/        # Example for single model and dataset
│   ├── scenario_b/        # One model per segment
│   └── scenario_c/        # Single model with segmentation column
├── reports/
│   ├── images/            # SHAP plots, cluster visualizations
│   ├── clusters/          # Cluster summary CSVs
│   └── *.docx             # Final validation reports
├── tanml/                 # Core source code
│   ├── cli/               # CLI logic
│   ├── engine/            # Orchestration and segmentation logic
│   ├── checks/            # Validation check classes
│   ├── check_runners/     # Check runners linked to the engine
│   ├── report/            # Report generation using docxtpl
│   └── utils/             # Data/model loaders and helpers
├── tests/                 # Unit tests (WIP or planned)
├── setup.py               # Package installer for pip
├── pyproject.toml         # Modern Python build metadata (PEP 518)
├── MANIFEST.in            # Files to include when packaging
└── README.md              # This documentation
```

## How to Use TanML

To use TanML effectively, follow these three main steps. This process is designed to be simple and intuitive, even for users without a programming background.

1. **Initialize the Configuration File:**
   Start by generating a YAML configuration file that tells TanML how to run the validation. You do this using a command-line instruction where you select one of the three supported validation scenarios:

   * Scenario A: For validating one model using a single dataset
   * Scenario B: For validating multiple models, each with its own segment-specific dataset
   * Scenario C: For validating a single model that needs to be run separately on different segments within one dataset

   You can also choose where the generated YAML file should be saved. If you don’t provide a location, it will be saved in your current directory with a default name.

2. **Fill in the Configuration File:**
   Once the YAML file is created, you’ll open it and fill in the necessary details:

   * Where your model file is located (e.g., a `.pkl` file for Python, or equivalent for SAS or R)
   * Where your cleaned dataset is saved
   * (Optional) Where your raw dataset is saved, if you want data quality comparisons
   * Which input features your model expects, and what the target column is
   * Where you want the final report to be saved (Word `.docx` format)

   This YAML file acts like a blueprint—TanML reads it and follows the instructions you provide to run all relevant validation checks.

3. **Run the Validation Process:**
   After the YAML file is completed, you will run the validation process. This will trigger TanML to:

   * Load your model and data
   * Perform all configured validation checks (like performance, SHAP explainability, stress testing, etc.)
   * Automatically generate a professional report with summaries, tables, and visuals

You can use TanML either through simple command-line instructions or directly in Python by calling its functions. Both methods achieve the same results. The command-line approach is ideal for repeatable, scriptable workflows, while the Python interface is useful for advanced users who want to integrate TanML into larger systems or notebooks.

TanML is controlled entirely through a YAML configuration file and a single CLI command. The configuration specifies the model, data, validation rules, and output paths.

### Scenario A: Single Model, Single Dataset

This is the simplest usage mode. You have one model file and one cleaned dataset.

**rules.yaml**

```yaml
model:
  features:
    - age
    - income
    - debt_to_income
    - credit_score
    - employment_length
  target: default_flag

paths:
  model: models/model.pkl
  cleaned_data: data/cleaned/cleaned.csv
  raw_data: data/raw/raw.csv

output:
  report_path: reports/validation_report.docx
```

**Run the validation:**

```bash
tanml validate --rules rules.yaml
```

This generates a `.docx` report along with SHAP plots and other artifacts.

### Scenario B: One Model per Segment

Use this if you have multiple segments, each with its own cleaned dataset and model.

**rules.yaml**

```yaml
segment:
  runs:
    segment_A:
      model: models/model_segment_A.pkl
      cleaned: data/cleaned/segment_A.csv
      raw: data/raw/segment_A.csv
      output_report: reports/report_segment_A.docx

    segment_B:
      model: models/model_segment_B.pkl
      cleaned: data/cleaned/segment_B.csv
      raw: data/raw/segment_B.csv
      output_report: reports/report_segment_B.docx

model:
  features:
    - age
    - income
    - debt_to_income
    - credit_score
    - employment_length
  target: default_flag
```

**Run the validation:**

```bash
tanml validate --rules rules.yaml
```

Each segment will be validated independently with its own output report.

### Scenario C: One Model with Segmentation Column

Use this if you have one dataset with a segmentation column (e.g., region, product type).

**rules.yaml**

```yaml
segment:
  column: segment_id  # This column will be used to split the data automatically

model:
  features:
    - age
    - income
    - debt_to_income
    - credit_score
    - employment_length
  target: default_flag

paths:
  model: models/credit_model.pkl
  cleaned_data: data/cleaned/combined.csv
  raw_data: data/raw/combined.csv

output:
  report_path: reports/report_{segment}.docx
```

**Run the validation:**

```bash
tanml validate --rules rules.yaml
```

This will automatically split the dataset by the `segment_id` column, apply the same model to each subset, and produce one report per segment.

## Output Artifacts

After a successful validation run, TanML generates a set of output files based on your configuration:

- **Validation Report (.docx):**  
  A professionally formatted Word document containing:
  - A summary of all validation checks
  - Tables for performance metrics, data quality, logistic regression coefficients, VIF, and more
  - SHAP-based feature importance visualizations (if enabled)
  - Segment-wise validation summaries (for Scenario B and C)

- **SHAP Visualizations:**  
  Summary bar plots and other SHAP outputs are saved in the `reports/images/` folder.

- **Input Cluster Coverage Charts and CSVs:**  
  If cluster coverage analysis is enabled, visual and tabular summaries are stored in:
  - `reports/images/` (cluster bar plots)
  - `reports/clusters/` (CSV summaries)

- **Logs and Intermediate Results:**  
  Optional debug or intermediate outputs (e.g., cleaned data snapshots or rule validation results) can be generated depending on configuration or verbosity level.

By default, all outputs are saved to the paths you define in the YAML configuration. Each segment (in Scenario B or C) will generate its own report file if multiple runs are triggered.

Extending TanML

TanML is designed with modularity in mind. Advanced users and developers can easily extend its capabilities by adding new validation checks or modifying report components. Here's how extension works:

Add a Custom Check:Create a new check class in the tanml/checks/ directory and implement the validation logic based on the BaseCheck interface.

Create a Check Runner:Add a corresponding runner in tanml/check_runners/. This file controls how the check is executed and connected to the engine.

Register the Check:Link your runner in the central registry inside the validation engine. The YAML config will trigger it automatically based on user-defined rules.

This modular structure ensures that domain-specific validations (e.g., industry regulations, fairness audits) can be added without modifying core logic.

## License and Citation

TanML is open-source under the MIT License.  
Copyright © 2025 Tanmay Sah and Dolly Sah.

You are free to use, modify, and distribute this software with appropriate attribution.

If you use TanML in your work or publication, please cite it as:

> Sah, T., & Sah, D. (2025). *TanML: Automated Model Validation Toolkit for Tabular Machine Learning*. GitHub. https://github.com/tdlabs-ai/tanml

📄 A machine-readable citation file (`CITATION.cff`) is included for use with citation tools and GitHub's “Cite this repository” button.
