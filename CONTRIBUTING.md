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

Add your tab to `tanml/ui/pages/evaluation.py`:

```python
# In the render function, add your tab
with tab_my_feature:
    from tanml.checks.my_feature import MyFeatureCheck
    
    check = MyFeatureCheck(model, X_train, X_test, y_train, y_test)
    result = check.run()
    
    st.subheader(result.name)
    st.write(f"Status: {result.status}")
    st.json(result.metrics)
```

### Step 3: Add Report Section (Optional)

```python
# tanml/ui/reports/sections/my_feature.py
from tanml.ui.reports.base import ReportSection, SectionRegistry

@SectionRegistry.register("evaluation")
class MyFeatureSection(ReportSection):
    name = "My Feature Analysis"
    order = 60  # Controls position in report
    
    def should_include(self, context) -> bool:
        return context.has("my_feature_results")
    
    def add_to_document(self, doc, context) -> None:
        doc.add_heading(self.name, level=2)
        results = context.get("my_feature_results")
        doc.add_paragraph(f"Score: {results['score']:.2f}")
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
streamlit run tanml/ui/app.py
```

## Questions?

Open an issue on the repository!
