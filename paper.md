---
title: 'TanML: Automated Model Validation Toolkit for Tabular Machine Learning Models'
tags:
  - Python
  - machine learning
  - model validation
  - model risk management
  - regulatory compliance
  - governance
authors:
  - name: Tanmay Sah
    orcid: 0009-0004-8583-2208
    affiliation: 1
  - name: Dolly Sah
    affiliation: 2
  - name: Harshul Jain
    affiliation: 3
  - name: Tanya Sah
    affiliation: 3
  - name: Kayden Jordan
    affiliation: 1
affiliations:
  - name: Harrisburg University of Science and Technology
    index: 1
  - name: University of Utah
    index: 2
  - name: New York University
    index: 3
date: 7 March 2026
bibliography: paper.bib
---

# Summary

TanML [@tanml_2026] is an open-source toolkit for validating machine learning models built on tabular (structured) data through a user friendly Streamlit interface. It supports an end-to-end workflow that includes data profiling, preprocessing, feature ranking, model development, evaluation, and export of editable audit ready reports. The toolkit is designed to help users validate models built with common Python ML libraries such as scikit-learn [@pedregosa2011scikit], XGBoost [@chen2016xgboost], statsmodels [@seabold2010statsmodels], LightGBM [@ke2017lightgbm], and CatBoost [@prokhorenkova2018catboost]. TanML is especially useful in regulated and documentation heavy settings [@sr11-7] because it combines model validation, explainability, drift and stress analysis, and report generation in a single local first workflow. It is intended for data scientists, quantitative analysts, and model risk stakeholders who need a practical way to review and document ML models trained on structured datasets without building custom validation pipelines from scratch.

# Statement of need

Machine learning models trained on structured, tabular data are widely used in high-stakes domains such as finance, insurance, and other regulated settings. In these environments, building a predictive model is only one part of the workflow, because practitioners must also assess data quality, model behavior, predictive performance, robustness, and interpretability while producing documentation that supports internal review, governance, and compliance processes. For example, banking institutions developing credit scoring or fraud detection models are required to produce comprehensive model validation reports documenting data integrity, performance benchmarking, and sensitivity analyses before regulatory approval [@sr11-7]. Similarly, insurance firms building claims forecasting models must demonstrate that their automated systems are fair, explainable, and robust to distributional shifts. In practice, these validation activities are often fragmented across notebooks, scripts, visualization libraries, and manually prepared reports, making workflows difficult to standardize, reproduce, and communicate, especially when results must be reviewed by stakeholders beyond the original model developer. TanML was developed to address this problem by providing a unified workflow for validating and documenting models trained on tabular data. Instead of requiring practitioners to assemble separate tools for analysis, evaluation, and reporting, TanML organizes these activities within a single interface intended to make validation outputs easier to reproduce, review, and communicate. The primary target audience includes data scientists, quantitative analysts, model validation teams, and other practitioners working in regulated or documentation heavy settings who need reviewable validation artifacts in addition to numerical metrics or exploratory analysis.


# State of the field

The open source ecosystem already includes several widely used tools that address important parts of the machine learning lifecycle for structured data. While tools exist to support individual components of this workflow, TanML brings all components together in a single software package. Data profiling packages such as YData Profiling [@clemente2023ydata] emphasize exploratory analysis and dataset understanding. Monitoring frameworks such as Evidently [@evidently] focus on drift detection and production observability. Validation libraries such as Deepchecks [@deepchecks] provide programmable checks for data and model quality. AutoML frameworks such as PyCaret [@pycaret] and AutoGluon [@erickson2020autogluon] focus on model training, comparison, and selection. These tools are valuable within their intended scope, but their primary objectives differ from TanML's integrated validation-and-documentation focus. Profiling tools are centered on dataset exploration rather than end-to-end model validation. Monitoring tools are designed mainly for observability and post-deployment drift analysis rather than pre-deployment validation workflows. Validation libraries provide useful checks, but they are generally oriented toward developer-driven testing rather than stakeholder facing validation workflows. AutoML systems improve modeling efficiency, but they do not primarily address governance, documentation, or audit-ready reporting requirements.TanML was developed to fill this gap by integrating validation-oriented tasks into a single workflow for structured-data machine learning. Rather than replacing existing tools individually, it combines model evaluation, explainability, drift analysis, and stakeholder-ready reporting in one UI-driven system. This also motivates the build-versus-contribute decision. TanML's scholarly contribution lies at the workflow level rather than in a single isolated method. The software was built as a standalone package because existing profiling, monitoring, testing, and AutoML tools do not fully address the combined need for tabular model validation and stakeholder-ready documentation.

+---------------+---------------+-------------------+---------------------+-------------------+
| Tool          | Primary Focus | Scope             | Output              | Drift Logic       |
+===============+===============+===================+=====================+===================+
| **TanML**     | Model Risk    | Data, Model       | Audit-ready         | PSI and KS        |
|               | Management    | , Governance      | report              |                   |
+---------------+---------------+-------------------+---------------------+-------------------+
| **Evidently   | Monitoring    | Data, Model       | Dashboards          | Statistical tests |
| AI**          |               |                   |                     |                   |
+---------------+---------------+-------------------+---------------------+-------------------+
| **Deepchecks**| Testing / CI  | Data, Model       | Reports             | Multiple methods  |
+---------------+---------------+-------------------+---------------------+-------------------+
| **AutoML**    | Model         | Model building    | Models              | N/A               |
|               | Training      |                   |                     |                   |
|               |               |                   |                     |                   |
+---------------+---------------+-------------------+---------------------+-------------------+
| **YData       | Data EDA      | Data only         | HTML report         | Warnings          |
| Profiling**   |               |                   |                     |                   |
+---------------+---------------+-------------------+---------------------+-------------------+
: Comparison of TanML with commonly used tools across primary focus, scope, outputs, and drift-analysis approach. \label{table:comparison}



# Software design

TanML was explicitly designed as a modular, privacy-first desktop application that balances analytical rigor with non-developer accessibility. By rejecting a traditional web-based frontend (e.g., React/JS) and instead managing the entire application flow locally in pure Python via **Streamlit** [@streamlit], the toolkit lowers the barrier to entry for quantitative analysts and model risk practitioners. This trade-off allows users to inspect, extend, and deploy the UI code directly alongside their statistical models without requiring dedicated web engineering teams. Importantly, the local-first architecture ensures that sensitive financial and regulatory data never leaves the user's machine. In domains such as banking and insurance, transmitting customer-level loan data or proprietary risk scores to cloud-hosted services may violate internal data governance policies or regulatory requirements. TanML avoids this risk entirely by keeping all computation and data in the user's local environment.

### Core Architecture

The first pillar is a centralized **Session State Manager** that handles data flow between modules without requiring a persistent database. This approach allows sensitive financial data to remain local to the user's environment, either held transiently in memory or stored in ephemeral local directories, rather than being transmitted to external hosted services. For example, when a practitioner uploads a loan-level dataset for credit model validation, the data persists only for the duration of the session and is never written to a remote store.

The second pillar is the **Model Registry**, a dynamic factory pattern (`tanml.models.registry`) that standardizes the instantiation of various estimators (XGBoost, CatBoost, LightGBM) with pre-configured hyperparameters. This decoupling allows researchers to easily inject novel models without altering the core validation engine UI.

The third pillar is the **Reporting Engine**, a `docx`-based generator that serializes analysis results into a structured document, mapping complex Python objects (such as **Matplotlib** [@Hunter:2007] figures and **Pandas** [@reback2020pandas] DataFrames) into native Open XML formats. The resulting Word document is immediately editable, allowing validation teams to annotate findings, append narrative commentary, and route the report through internal review processes without reformatting.

### Modular Workflow

The application is divided into distinct execution layers, designed to mirror the standard Model Risk Management lifecycle.

The **Data Profiling and Preprocessing** layer handles exploratory analysis, automated cleaning, high cardinality encoding, and missing value imputation. For instance, a risk analyst examining a credit scoring dataset can review distributional summaries, identify missing value patterns across borrower attributes, and apply encoding strategies for categorical features such as loan purpose or geographic region, all within a guided interface before any model is trained.

The **Feature Power Ranking** layer assesses predictive power and statistical significance (p-values) before modeling. This allows practitioners to identify which features contribute meaningfully to the target variable and to document the rationale for feature inclusion or exclusion, a common requirement in model validation reports.

The **Model Development** layer facilitates champion-challenger comparisons using K-Fold Cross-Validation. Users can train and compare multiple estimators side-by-side, selecting a champion model while retaining challenger results for documentation purposes.

The **Evaluation Suite** is the core validation engine. It calculates Population Stability Index (PSI) for drift analysis, tree-based SHAP values [@lundberg2017unified] for explainability, and segmented performance metrics. For example, a model risk team validating an insurance claims model can use PSI to assess whether the production data has drifted from the training distribution, examine SHAP plots to verify that feature contributions align with domain expectations, and review performance breakdowns across customer segments to detect potential fairness or stability issues. These outputs are then serialized directly into the generated validation report.

To illustrate a complete workflow: a quantitative analyst at a financial institution receives a challenger credit model and its associated dataset. Using TanML, the analyst profiles the data to check for quality issues, ranks features by predictive power, retrains the model under cross-validation to verify reported performance, runs drift and SHAP analyses against a holdout or production sample, and exports a structured Word report. This report can then be circulated to model governance committees and regulators with minimal additional preparation.

![High-level modular workflow of TanML. \label{fig:arch}](architecture.png)

# Research impact statement

TanML has accumulated over 2,500 downloads on PyPI since its initial release in July 2025 and is scheduled to be presented at the North Carolina State University Master in Financial Mathematics Workshop on Quantitative Finance in April 2026. The toolkit is packaged and installable via `pip install tanml`, archived on Zenodo with a persistent DOI [@tanml_2026], and documented through a published MkDocs site. The repository includes an automated test suite run via GitHub Actions CI, a `CONTRIBUTING.md` guide, a code of conduct, and an open issue tracker to support community contributions. Reproducible demo datasets and example workflows are provided in the repository so that new users can evaluate the toolkit's end-to-end validation capabilities on representative tabular data without requiring proprietary datasets. The modular architecture, described in the Software Design section, provides a clear extension pathway for contributors seeking to add new model families, evaluation metrics, or report templates.

# AI usage disclosure

Portions of the `TanML` codebase, including specific unit tests and documentation templates, were refactored with the assistance of Large Language Models (LLMs). The human maintainers have reviewed and verified all AI-generated contributions to ensure technical accuracy.

# Acknowledgements

We acknowledge the valuable feedback from the pyOpenSci editor and reviewers, whose insights significantly shaped and improved the toolkit. We also express our appreciation to the maintainers and contributors of the open-source libraries that TanML builds upon, including NumPy, pandas, SciPy, Numba, scikit-learn, statsmodels, XGBoost, matplotlib, seaborn, Pillow, python-docx, Streamlit, openpyxl, pyarrow, and pyreadstat. In addition, we are grateful to all those who shared ideas, suggestions, and feature requests that helped inform the development of TanML.

# References
