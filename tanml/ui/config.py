# tanml/ui/config.py
"""
UI configuration and styling for TanML.

Provides centralized styling and configuration constants.
"""

from __future__ import annotations

import streamlit as st


# =============================================================================
# App Configuration
# =============================================================================

APP_TITLE = "TanML"
APP_ICON = "🧪"
APP_DESCRIPTION = "Industrial-Grade Model Validation"


# =============================================================================
# Page Configuration
# =============================================================================

def configure_page(
    title: str = APP_TITLE,
    icon: str = APP_ICON,
    layout: str = "wide",
) -> None:
    """
    Configure Streamlit page settings.
    
    Should be called once at app startup.
    
    Args:
        title: Page title
        icon: Page icon (emoji)
        layout: Page layout ("wide" or "centered")
    """
    st.set_page_config(
        page_title=title,
        page_icon=icon,
        layout=layout,
        initial_sidebar_state="expanded",
    )


# =============================================================================
# CSS Styling
# =============================================================================

def load_css() -> None:
    """
    Load custom CSS styling for TanML UI.
    
    Should be called once at app startup after configure_page().
    """
    st.markdown(get_custom_css(), unsafe_allow_html=True)


def get_custom_css() -> str:
    """
    Get the custom CSS for TanML UI.
    
    Returns:
        CSS string
    """
    return """
    <style>
    /* KPI Styling */
    .tanml-kpi-label {
        font-size: 0.80rem;
        opacity: 0.8;
        white-space: nowrap;
        height: 20px;
        display: flex;
        align-items: flex-end;
    }
    
    .tanml-kpi-value {
        font-size: 1.6rem;
        font-weight: 700;
        line-height: 1;
        margin-top: 4px;
    }
    
    /* Card styling */
    .tanml-card {
        background: white;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    
    /* Section headers */
    .tanml-section-header {
        font-size: 1.25rem;
        font-weight: 600;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e5e7eb;
    }
    
    /* Status badges */
    .tanml-badge-ok {
        background-color: #dcfce7;
        color: #166534;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.75rem;
        font-weight: 500;
    }
    
    .tanml-badge-warning {
        background-color: #fef3c7;
        color: #92400e;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.75rem;
        font-weight: 500;
    }
    
    .tanml-badge-error {
        background-color: #fecaca;
        color: #991b1b;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.75rem;
        font-weight: 500;
    }
    
    /* Metric tables */
    .tanml-metrics-table td {
        padding: 0.5rem;
        border-bottom: 1px solid #e5e7eb;
    }
    
    .tanml-metrics-table th {
        padding: 0.5rem;
        background-color: #f9fafb;
        font-weight: 600;
        text-align: left;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #888;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #555;
    }
    </style>
    """


# =============================================================================
# Theme Colors
# =============================================================================

COLORS = {
    "primary": "#2563eb",
    "secondary": "#7c3aed",
    "success": "#10b981",
    "warning": "#f59e0b",
    "danger": "#ef4444",
    "info": "#06b6d4",
    "neutral": "#6b7280",
}


def get_status_color(status: str) -> str:
    """
    Get color for a status value.
    
    Args:
        status: "ok", "warning", "error", or "skipped"
        
    Returns:
        Hex color code
    """
    status_colors = {
        "ok": COLORS["success"],
        "warning": COLORS["warning"],
        "error": COLORS["danger"],
        "skipped": COLORS["neutral"],
    }
    return status_colors.get(status, COLORS["neutral"])


def get_status_badge(status: str, text: str = None) -> str:
    """
    Get HTML for a status badge.
    
    Args:
        status: "ok", "warning", "error", or "skipped"
        text: Optional custom text (defaults to status)
        
    Returns:
        HTML string for the badge
    """
    text = text or status.upper()
    css_class = f"tanml-badge-{status}"
    return f'<span class="{css_class}">{text}</span>'
