# Contributing to TanML

Thank you for your interest in contributing to TanML! This guide will help you get started.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/tdlabs-ai/tanml.git
cd tanml

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/
```

## Architecture Overview

TanML follows SOLID principles with a modular architecture:

```
tanml/
├── core/              # Core abstractions (protocols, context, exceptions)
├── config/            # Configuration management
├── check_runners/     # Validation check implementations
├── checks/            # Check logic (separated from runners)
├── engine/            # Validation orchestration
├── report/            # Report generation
├── models/            # Model registry
├── utils/             # Utility functions
└── ui/                # Streamlit interface
```

## Adding a New Validation Check

This is the most common contribution - adding new validation checks.

### Step 1: Create the Check Runner

Create a new file in `tanml/check_runners/`:

```python
# tanml/check_runners/my_check_runner.py
"""
My custom check runner.

This check does [describe what it does].
"""

from typing import Any, Dict
from tanml.check_runners.base_runner import BaseCheckRunner


class MyCheckRunner(BaseCheckRunner):
    """
    Runner for my custom check.
    
    Configuration Options:
        option1: Description (default: value)
        option2: Description (default: value)
        
    Output:
        - result1: Description
        - result2: Description
    """
    
    @property
    def name(self) -> str:
        return "MyCheck"
    
    def execute(self) -> Dict[str, Any]:
        """
        Run the check logic.
        
        Returns:
            Dictionary containing check results
        """
        # Access configuration
        option1 = self.get_config_value("option1", default_value)
        
        # Access data
        X_train = self.context.X_train
        y_train = self.context.y_train
        cleaned_df = self.context.cleaned_df
        
        # Get output directory for artifacts
        output_dir = self.get_output_dir()
        
        # Your check logic here
        result = {"status": "ok", "my_metric": 0.95}
        
        return result


# Legacy compatibility function (optional but recommended)
def run_my_check(model, X_train, X_test, y_train, y_test, 
                 rule_config, cleaned_df, *args, **kwargs):
    """Legacy function interface."""
    from tanml.core.context import CheckContext
    
    context = CheckContext(
        model=model,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        config=rule_config,
        cleaned_df=cleaned_df,
    )
    
    runner = MyCheckRunner(context)
    return runner.run()
```

### Step 2: Register the Check

Add your check to `tanml/engine/check_agent_registry.py`:

```python
from tanml.check_runners.my_check_runner import MyCheckRunner, run_my_check

# Add to both registries
CHECK_RUNNER_REGISTRY["MyCheck"] = run_my_check
CHECK_RUNNER_CLASSES["MyCheck"] = MyCheckRunner
```

Or use the registration function:

```python
from tanml import register_check
from tanml.check_runners.my_check_runner import MyCheckRunner, run_my_check

register_check("MyCheck", MyCheckRunner, run_my_check)
```

### Step 3: Add Tests

Create tests in `tests/check_runners/test_my_check_runner.py`:

```python
import pytest
import pandas as pd
from tanml.core.context import CheckContext
from tanml.check_runners.my_check_runner import MyCheckRunner


@pytest.fixture
def sample_context():
    # Create sample data
    X_train = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    y_train = pd.Series([0, 1, 0])
    X_test = pd.DataFrame({"a": [7, 8], "b": [9, 10]})
    y_test = pd.Series([1, 0])
    
    return CheckContext(
        model=None,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        config={"MyCheck": {"enabled": True}},
        cleaned_df=X_train,
    )


def test_my_check_runs(sample_context):
    runner = MyCheckRunner(sample_context)
    result = runner.run()
    
    assert result is not None
    assert "status" in result


def test_my_check_disabled(sample_context):
    context = CheckContext(
        model=sample_context.model,
        X_train=sample_context.X_train,
        X_test=sample_context.X_test,
        y_train=sample_context.y_train,
        y_test=sample_context.y_test,
        config={"MyCheck": {"enabled": False}},
        cleaned_df=sample_context.cleaned_df,
    )
    
    runner = MyCheckRunner(context)
    result = runner.run()
    
    assert result is None  # Disabled checks return None
```

## Adding Report Sections

To add a new section to the generated reports:

1. Add your section logic in `tanml/report/report_builder.py`
2. Update the template if using placeholders

## Code Style

- Follow PEP 8 guidelines
- Use type hints for function signatures
- Add docstrings with Args, Returns, and Examples
- Keep functions focused (Single Responsibility Principle)

## Submitting Changes

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-check`
3. Make your changes with tests
4. Run the test suite: `pytest tests/`
5. Submit a pull request

## Questions?

Open an issue on GitHub or reach out to the maintainers.
