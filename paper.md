---
title: 'TanML: A Secure, Automated Validator for Tabular Machine Learning'
tags:
  - Python
  - machine learning
  - model risk management
  - validation
  - finance
  - explainability
  - drift detection
authors:
  - name: Tanmay Sah
    orcid: 0009-0004-8583-2208
    affiliation: 1
  - name: Dolly Sah
    affiliation: 1
affiliations:
  - name: tdlabs.ai, San Francisco, CA, USA
    index: 1
date: 22 January 2026
bibliography: paper.bib
---

# Summary

Financial institutions and other regulated industries rely on robust Model Risk Management (MRM) frameworks to ensure the safety and accountability of machine learning (ML) models. While these standards (e.g., SR 11-7) are critical, traditional validation workflows are often manual, time-consuming, and prone to human error.

`TanML` is a privacy-first, automated validation framework designed specifically for high-stakes applications. It integrates data profiling, automated baseline model training (XGBoost, CatBoost, LightGBM), and a comprehensive validation suite into a unified Python library with an interactive local dashboard. Its primary contribution is the "last-mile" automation of validation, converting complex computational analyses into standardized, regulatory-grade audit reports (Word .docx format).

# Statement of Need

The "innovation paradox" in finance means that while sophisticated ML models are technically available, their deployment is often delayed by lengthy manual validation cycles. Validation teams frequently spend weeks manually calculating stability metrics like the Population Stability Index (PSI), generating SHAP-based interpretations [@shap], and copy-pasting charts into documents.

`TanML` addresses this need by providing:
1.  **Automation**: A standardized pipeline from data ingestion to document generation.
2.  **Privacy**: A local-only execution model ensuring no sensitive financial data leaves the user's environment.
3.  **Governance**: Direct mapping of diagnostic metrics to the requirements of the Federal Reserve’s SR 11-7 guidance.

# State of the Field

Existing platforms like **PyCaret** [@pycaret] and **AutoGluon** provide powerful capabilities for the model training phase, but lack the integrated "validation-to-report" workflow needed for compliance. Monitoring platforms like **Evidently AI** [@evidently] offer robust post-deployment drift detection, but are not designed for the exhaustive pre-deployment audits that `TanML` facilitates. `TanML` occupies a distinct niche by acting as a bridge between specialized training tools and final regulatory approval.

# Research Impact

`TanML` democratizes "Tier-1 Bank" quality model validation, enabling smaller fintechs and insurance firms to meet rigorous compliance standards without massive dedicated MRM departments.

# Acknowledgements

We acknowledge the contributors and maintainers of the underlying scientific Python ecosystem, particularly the developers of Streamlit, Scikit-learn, and the various gradient boosting libraries that powered the benchmarking engine.

# References
