# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - Unreleased

### Added
- Feature A: Automated PyPI releases via Tag-triggered GitHub Actions.
- Feature B: Drag-and-drop file upload capability in the Data Profiling UI.
- Feature C: Logarithmic Scale toggle for Data Profiling graphs.
- Feature D: Integrated Train-Test Split step in the Preprocessing pipeline.
- Feature E: Statsmodels integration for p-values and statistical significance.
- **Phase 3 Core Analytics**: Integrated Tree-based SHAP values for explainability.
- **JOSS Publication**: Added a programmatic architecture diagram and detailed comparison matrices for the pyOpenSci/JOSS submission.
- New `dev/v0.2.0` development branch strategy for safe feature iteration.

### Changed
- Updated CI workflow (`ci.yml`) to support automated testing for development branches.
- Modernized CI dependencies and testing logic for faster, more reliable builds.

### Fixed
- **Infrastructure**: Pinned `numba>=0.61.0` to resolve installation failures on Python 3.10.
- **Reporting**: Initiated refactor of Word report generators to improve maintainability.
- Added Python 3.13 to the official testing matrix.

## [0.1.10] - 2026-03-07
- Initial baseline for pyOpenSci submission.
