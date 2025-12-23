# Contributing to TanML

Welcome! TanML uses a modular, extensible architecture designed to make it easy to add new features.

## Architecture Overview

```
tanml/
├── checks/           # Analysis checks (SHAP, Stress Test)
│   ├── base.py       # BaseCheck, CheckResult, CheckRegistry
│   └── ...
│
├── analysis/         # Reusable business logic
│   ├── drift.py      # PSI/KS drift analysis
│   ├── clustering.py # Cluster coverage
│   └── correlation.py # VIF, correlation
│
├── ui/
│   ├── pages/        # UI pages
│   ├── reports/      # Report generation
│   │   ├── base.py   # ReportSection, SectionRegistry
│   │   └── ...
│   └── components/   # Reusable UI components
│
└── models/           # Model registry
```

## Adding a New Analysis Feature

### Step 1: Create the Check (Business Logic)

```python
# tanml/checks/my_feature.py
from tanml.checks.base import BaseCheck, CheckResult, CheckRegistry

@CheckRegistry.register
class MyFeatureCheck(BaseCheck):
    name = "My Feature Check"
    
    def run(self) -> CheckResult:
        # Your analysis logic
        score = self._calculate_score()
        
        return CheckResult(
            name=self.name,
            status="pass" if score > 0.8 else "warning",
            metrics={"score": score},
            summary=f"Score: {score:.2f}",
        )
    
    def _calculate_score(self):
        # Use self.X_train, self.X_test, etc.
        return 0.95
```

### Step 2: Add UI Tab in Evaluation Page

Instead of modifying a monolithic file, simply create a **new file** in `tanml/ui/pages/evaluation/tabs/` (e.g., `my_tab.py`).

1. **Create File**: `tanml/ui/pages/evaluation/tabs/my_feature.py`
2. **Use Decorator**: Register your tab.
3. **Implement Render**:

```python
# tanml/ui/pages/evaluation/tabs/my_feature.py
import streamlit as st
from tanml.ui.pages.evaluation.tabs import register_tab, TabContext

@register_tab(name="My Feature", order=50, key="tab_my_feature")
def render(context: TabContext):
    st.header("My Feature Analysis")
    
    # Access standardized data via context
    st.write(f"Analyzing {context.task_type} model...")
    st.dataframe(context.X_test.head())
    
    # Save results to context if needed for report
    context.results["my_feature"] = {"score": 0.99}
```

The system will **automatically discover** this file and add the tab to the Evaluation page!

### Step 3: Add Report Section

Currently, report generation is handled in `tanml/ui/reports/generators.py`.

1. Open `tanml/ui/reports/generators.py`.
2. Locate `_generate_eval_report_docx`.
3. Add logic to extract your data from `eval_data` (which comes from `context.results`) and add paragraphs/tables to the `doc`.

```python
    # In _generate_eval_report_docx
    my_data = buf.get("my_feature")
    if my_data:
        doc.add_heading('My Feature Analysis', level=2)
        doc.add_paragraph(f"Score: {my_data['score']}")
```

## Using Analysis Modules

The `tanml/analysis/` module contains reusable business logic:

```python
from tanml.analysis import calculate_psi, calculate_vif, analyze_drift

# Calculate drift
drift_results = analyze_drift(train_df, test_df)

# Calculate VIF
vif_results = calculate_vif(df, features=["col1", "col2"])
```

## Code Style

- Use type hints
- Add docstrings with examples
- Follow existing patterns in similar files
- Test your changes before submitting

## Running the App

```bash
# Install in development mode
pip install -e .

# Run the UI
tanml ui
```

## Questions?

Open an issue on the repository!
