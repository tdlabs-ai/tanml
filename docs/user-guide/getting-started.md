# Getting Started

This guide walks you through the end-to-end lifecycle of a model validation project using TanML.

## Prerequisites

TanML requires **Python 3.10 or higher**. We strongly recommend using a virtual environment:

```bash
# Create and activate environment
python -m venv .venv
source .venv/bin/activate  # Mac/Linux
# .venv\Scripts\activate   # Windows

# Install TanML
pip install tanml
```

## The Validation Lifecycle

Launch the dashboard to begin:

```bash
tanml ui
```

### 1. Data Ingestion
The first step is loading your data. TanML supports **CSV** and **Parquet** formats.
- **Upload**: Drop your training and testing datasets into the upload area.
- **Target Selection**: Identify the column you want to predict. TanML will automatically infer if it is a **Classification** or **Regression** task.

### 2. Data Profiling
Before modeling, you must understand your distributions.
- Navigate to the **Profiling** tab to see descriptive statistics and histograms for every feature.
- Look for outliers, data quality issues, or unexpected shifts that might impact model performance.

### 3. Preprocessing
TanML simplifies data cleaning for tabular ML.
- **Null Handling**: View missing value counts and apply automated imputation (Mean/Median for numeric, Mode for categorical).
- **Encoding**: Automatically convert categorical text into model-ready numeric formats.

### 4. Feature Power Ranking
Identify which features actually drive your model.
- The **Ranking** tab uses mutual information and correlation analysis to rank features by their predictive strength.
- Use this to prune "noise" features and focus your validation on the most influential variables.

### 5. Benchmark & Evaluation
Train a high-performance benchmark (XGBoost/LightGBM) to establish a baseline.
- **Performance**: View ROC curves, Precision-Recall charts, and Confusion Matrices.
- **Drift**: Calculate the **Population Stability Index (PSI)** to see if your test data has shifted away from your training data.
- **XAI**: Generate **SHAP** beeswarm plots to see *why* the model is making specific decisions.

### 6. Reporting
The final and most critical step for compliance.
- Click **Generate Report** in the sidebar.
- TanML programmatically builds a professional **.docx** file containing all charts, metrics, and data summaries.
- This report is ready-to-edit for your Model Risk Management (MRM) or audit folder.

---

## Pro Tips for Teams

### Remote Access
Running TanML on a powerful office server? Use the public flag:
```bash
tanml ui --public --port 8501
```
*Note: Ensure you are on a secure VPN/network when using `--public`.*

### Headless Mode
Running on a server without a monitor?
```bash
tanml ui --headless
```
This starts the backend without trying to open a local browser. You can then access it from your laptop using the server's IP address.
