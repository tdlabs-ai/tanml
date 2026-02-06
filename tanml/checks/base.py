# tanml/checks/base.py
"""
Base classes for TanML checks.

This module provides the foundation for creating extensible, maintainable checks.
Contributors can create new checks by inheriting from BaseCheck.

Example:
    from tanml.checks.base import BaseCheck

    class MyCustomCheck(BaseCheck):
        name = "My Custom Check"
        description = "Performs custom analysis on the model"

        def run(self) -> dict:
            # Your analysis logic here
            results = self._analyze()
            return {
                "status": "pass" if results["score"] > 0.8 else "warning",
                "metrics": results,
                "plots": [fig1, fig2],
            }
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    pass


@dataclass
class CheckResult:
    """
    Standardized result from a check.

    Attributes:
        name: Name of the check
        status: "pass", "warning", or "fail"
        metrics: Dictionary of computed metrics
        summary: Human-readable summary
        plots: List of matplotlib figures (for UI display)
        report_data: Data formatted for report generation
    """

    name: str
    status: str  # "pass", "warning", "fail"
    metrics: dict[str, Any] = field(default_factory=dict)
    summary: str = ""
    plots: list[Any] = field(default_factory=list)
    report_data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "status": self.status,
            "metrics": self.metrics,
            "summary": self.summary,
            "report_data": self.report_data,
        }


class BaseCheck(ABC):
    """
    Abstract base class for all TanML checks.

    Subclasses must implement:
        - name: Class attribute with check name
        - run(): Method that performs the analysis and returns CheckResult

    Example:
        class DriftCheck(BaseCheck):
            name = "Feature Drift Analysis"

            def run(self) -> CheckResult:
                psi_scores = self._calculate_psi()
                return CheckResult(
                    name=self.name,
                    status="warning" if any(psi > 0.1 for psi in psi_scores.values()) else "pass",
                    metrics={"psi": psi_scores},
                    summary=f"Analyzed {len(psi_scores)} features for drift",
                )
    """

    name: str = "Base Check"
    description: str = ""

    def __init__(
        self,
        model: Any,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        config: dict[str, Any] | None = None,
    ):
        """
        Initialize the check with model and data.

        Args:
            model: Trained model with predict/predict_proba methods
            X_train: Training features
            X_test: Test features
            y_train: Training labels
            y_test: Test labels
            config: Optional configuration dictionary
        """
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.config = config or {}

    @abstractmethod
    def run(self) -> CheckResult:
        """
        Execute the check and return results.

        Returns:
            CheckResult with status, metrics, and optional plots
        """
        pass

    def _get_config(self, key: str, default: Any = None) -> Any:
        """Get a configuration value with a default."""
        return self.config.get(key, default)

    def _infer_task_type(self) -> str:
        """Infer whether this is classification or regression."""
        unique_values = len(np.unique(self.y_train))
        if unique_values <= 10:
            return "classification"
        return "regression"


class CheckRegistry:
    """
    Registry for available checks.

    Allows dynamic registration and discovery of checks.

    Example:
        # Register a check
        CheckRegistry.register(DriftCheck)

        # Get all registered checks
        for check_cls in CheckRegistry.get_all():
            result = check_cls(model, X_train, X_test, y_train, y_test).run()
    """

    _checks: dict[str, type] = {}

    @classmethod
    def register(cls, check_class: type) -> type:
        """Register a check class. Can be used as decorator."""
        cls._checks[check_class.name] = check_class
        return check_class

    @classmethod
    def get(cls, name: str) -> type | None:
        """Get a check class by name."""
        return cls._checks.get(name)

    @classmethod
    def get_all(cls) -> list[type]:
        """Get all registered check classes."""
        return list(cls._checks.values())

    @classmethod
    def list_names(cls) -> list[str]:
        """List all registered check names."""
        return list(cls._checks.keys())
