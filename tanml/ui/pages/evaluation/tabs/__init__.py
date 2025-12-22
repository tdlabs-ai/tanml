# tanml/ui/pages/evaluation/tabs/__init__.py
"""
Tab Registry for Evaluation Page.

This module provides a registry pattern for evaluation tabs.
Contributors can add new tabs by creating a file in this folder
and using the @register_tab decorator.

Example:
    # tanml/ui/pages/evaluation/tabs/my_tab.py
    from tanml.ui.pages.evaluation.tabs import register_tab
    
    @register_tab(name="My Analysis", order=80)
    def render(context):
        import streamlit as st
        st.markdown("### My Analysis")
        # Your tab content here
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional
import importlib
import pkgutil


@dataclass
class TabContext:
    """
    Context passed to each tab render function.
    
    Contains all the data needed for analysis.
    """
    model: Any
    X_train: Any
    X_test: Any
    y_train: Any
    y_test: Any
    y_pred_train: Any
    y_pred_test: Any
    task_type: str
    features: List[str]
    target: str
    
    # Optional
    y_prob_train: Optional[Any] = None
    y_prob_test: Optional[Any] = None
    
    # For storing results (passed to report)
    results: Dict[str, Any] = None
    images: Dict[str, bytes] = None
    
    def __post_init__(self):
        if self.results is None:
            self.results = {}
        if self.images is None:
            self.images = {}


@dataclass
class TabDefinition:
    """Definition of a registered tab."""
    name: str           # Display name in UI
    render_fn: Callable # Function that renders the tab
    order: int = 100    # Display order (lower = first)
    key: str = ""       # Unique key for Streamlit


# Global registry
_TAB_REGISTRY: List[TabDefinition] = []


def register_tab(name: str, order: int = 100, key: str = ""):
    """
    Decorator to register a tab render function.
    
    Args:
        name: Display name for the tab
        order: Sort order (lower = appears first)
        key: Optional unique key for Streamlit widgets
        
    Example:
        @register_tab(name="Drift Analysis", order=30)
        def render(context):
            st.markdown("### Drift")
            ...
    """
    def decorator(fn: Callable):
        tab_key = key or fn.__name__
        _TAB_REGISTRY.append(TabDefinition(
            name=name,
            render_fn=fn,
            order=order,
            key=tab_key,
        ))
        return fn
    return decorator


def get_registered_tabs() -> List[TabDefinition]:
    """Get all registered tabs sorted by order."""
    return sorted(_TAB_REGISTRY, key=lambda t: t.order)


def discover_tabs():
    """
    Auto-discover and import all tab modules in this package.
    
    This imports all Python files in the tabs/ folder,
    which causes their @register_tab decorators to run.
    """
    import tanml.ui.pages.evaluation.tabs as tabs_package
    
    for importer, modname, ispkg in pkgutil.iter_modules(tabs_package.__path__):
        if not modname.startswith('_'):  # Skip __init__, etc.
            importlib.import_module(f'tanml.ui.pages.evaluation.tabs.{modname}')


def clear_registry():
    """Clear all registered tabs (useful for testing)."""
    global _TAB_REGISTRY
    _TAB_REGISTRY = []
