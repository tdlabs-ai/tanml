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
    
    /* Card styling - Merged Base */
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
    .tanml-badge-ok { background-color: #dcfce7; color: #166534; padding: 0.25rem 0.5rem; border-radius: 0.25rem; font-size: 0.75rem; font-weight: 500; }
    .tanml-badge-warning { background-color: #fef3c7; color: #92400e; padding: 0.25rem 0.5rem; border-radius: 0.25rem; font-size: 0.75rem; font-weight: 500; }
    .tanml-badge-error { background-color: #fecaca; color: #991b1b; padding: 0.25rem 0.5rem; border-radius: 0.25rem; font-size: 0.75rem; font-weight: 500; }
    
    /* Metric tables */
    .tanml-metrics-table td { padding: 0.5rem; border-bottom: 1px solid #e5e7eb; }
    .tanml-metrics-table th { padding: 0.5rem; background-color: #f9fafb; font-weight: 600; text-align: left; }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: #f1f1f1; }
    ::-webkit-scrollbar-thumb { background: #888; border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: #555; }
    
    /* --- APP SPECIFIC STYLES (Merged from app.py) --- */
    
    /* Global Background */
    .stApp {
        background-color: #f8fafc; /* Slate-50 */
        color: #0f172a; /* Slate-900 - Force dark text on light bg for Dark Mode users */
    }
    
    /* Layout Fixes */
    .block-container {
        padding-top: 1rem !important; 
        padding-bottom: 2rem !important;
        margin-top: 0px !important;
    }
    
    header[data-testid="stHeader"] {
        background-color: transparent;
        height: 3rem;
    }
    
    /* Typography */
    html, body, [class*="css"] {
        font-family: 'Source Sans Pro', sans-serif;
    }
    h1, h2, h3 {
        color: #1e293b;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin-top: 0px !important; 
        padding-top: 0px !important;
    }
    h4 {
        color: #334155;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    
    /* Navbar / Sidebar items */
    .stRadio label {
        font-size: 15px !important;
        padding-top: 4px;
        padding-bottom: 4px;
    }
    
    /* Primary Action Buttons */
    div.stButton > button:first-child {
        background-color: #2563eb;
        color: white;
        border: none;
        border-radius: 6px;
        font-weight: 600;
        padding: 0.5rem 1rem;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        transition: all 0.2s;
    }
    div.stButton > button:first-child:hover {
        background-color: #1d4ed8;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    div.stButton > button:first-child:disabled {
        background-color: #cbd5e1;
        color: #94a3b8;
        cursor: not-allowed;
    }
    
    /* File Uploader Container */
    [data-testid='stFileUploader'] {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
    }
    
    /* Enhanced Card Styling */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: white;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        padding: 2rem;
    }
    
    /* Task Badges */
    .task-badge {
        background-color: #dbeafe;
        color: #1e40af;
        padding: 4px 12px;
        border-radius: 16px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        border: 1px solid #bfdbfe;
    }
    
    /* Success/Info boxes - Make them flatter */
    .stAlert {
        border-radius: 6px;
        border: none;
        box-shadow: 0 1px 2px 0 rgba(0,0,0,0.05);
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


def get_status_badge(status: str, text: str | None = None) -> str:
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
