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

### Step 1: Understand Your Data (Data Profiling)
Start by uploading your raw dataset (**CSV** or **Parquet** format). Once you select your **Target Variable**, TanML will automatically analyze your data quality onscreen, identifying missing values, highlighting outliers, and visualizing the distribution of each variable to provide a baseline understanding of your data.

### Step 2: Clean & Split (Data Preprocessing)
Raw data often requires preparation before modeling. Upload your dataset to apply imputation strategies for missing values and encode text categories. Then, use the built-in **Data Splitter** at the bottom of the page to divide your cleaned dataset into a dedicated *Training Set* (for model learning) and an independent *Testing Set* (for final validation). You can export and download these datasets for the next steps.

### Step 3: Analyze Features (Feature Power Ranking)
Upload your newly created *Training Set*. TanML will mathematically rank your features onscreen to identify which ones show predictive power for your target variable, and which ones are statistically insignificant. Use these insights to drop low-ranking features and streamline your dataset. **Click 'Generate Feature Report'** to download a detailed report of your variables.

### Step 4: Build & Iterate (Model Development)
Upload your optimized *Training Set* and experiment with different algorithms, from classical regressions to tree-based models. TanML handles Cross-Validation in the background to ensure your model learns the underlying patterns, helping you identify the best performing configuration. **Click 'Generate Development Report'** to download a comprehensive summary of your model's performance.

### Step 5: Validate Performance (Model Evaluation)
Once you have selected a model configuration from Step 4, upload your original *Training Set* alongside your unseen *Testing Set*. TanML will evaluate your final model onscreen to check how well it generalizes to unseen data, monitor for Data Drift, and generate Explainability (SHAP) plots to help interpret the results. **Click 'Generate Evaluation Report'** to download a complete, audit-ready document proving your final model's real-world readiness.

---

## Advanced Configuration

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
