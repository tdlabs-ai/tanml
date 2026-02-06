# tanml/ui/components/metrics.py
"""
Metric display components for TanML UI.

Provides consistent styling for KPIs and performance metrics.
"""

from __future__ import annotations

from typing import Any

import streamlit as st


def metric_no_trunc(
    label: str,
    value: Any,
    delta: Any | None = None,
    delta_color: str = "normal",
    help: str | None = None,
) -> None:
    """
    Display a metric without truncation.

    Standard st.metric truncates long values. This version preserves
    the full value display.

    Args:
        label: Metric label
        value: Metric value (any type, will be converted to string)
        delta: Optional delta value
        delta_color: "normal", "inverse", or "off"
        help: Optional tooltip text
    """
    st.markdown(
        f"""
    <div class="tanml-kpi-label">{label}</div>
    <div class="tanml-kpi-value">{value}</div>
    """,
        unsafe_allow_html=True,
    )

    if delta is not None:
        if delta_color == "inverse":
            color = "green" if float(delta) < 0 else "red"
        elif delta_color == "off":
            color = "gray"
        else:
            color = "green" if float(delta) > 0 else "red"
        st.markdown(
            f'<span style="color:{color}; font-size:0.8rem;">Δ {delta}</span>',
            unsafe_allow_html=True,
        )


def metric_card(
    title: str,
    value: Any,
    icon: str = "📊",
    color: str = "#2563eb",
    subtitle: str | None = None,
) -> None:
    """
    Display a styled metric card.

    Args:
        title: Card title
        value: Primary value to display
        icon: Emoji or icon character
        color: Accent color (hex)
        subtitle: Optional subtitle text
    """
    st.markdown(
        f"""
    <div style="
        background: linear-gradient(135deg, {color}15 0%, {color}05 100%);
        border-left: 4px solid {color};
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
    ">
        <div style="display: flex; align-items: center; gap: 0.5rem;">
            <span style="font-size: 1.5rem;">{icon}</span>
            <span style="font-size: 0.85rem; color: #666;">{title}</span>
        </div>
        <div style="font-size: 1.8rem; font-weight: 700; color: {color}; margin-top: 0.25rem;">
            {value}
        </div>
        {f'<div style="font-size: 0.75rem; color: #888; margin-top: 0.25rem;">{subtitle}</div>' if subtitle else ""}
    </div>
    """,
        unsafe_allow_html=True,
    )


def kpi_row(
    metrics: list[tuple[str, Any, str | None]],
    columns: int | None = None,
) -> None:
    """
    Display a row of KPI metrics.

    Args:
        metrics: List of (label, value, icon) tuples
        columns: Number of columns (default: len(metrics))

    Example:
        kpi_row([
            ("ROC AUC", 0.92, "🎯"),
            ("Accuracy", 0.88, "✅"),
            ("F1 Score", 0.85, "📊"),
        ])
    """
    n_cols = columns or len(metrics)
    cols = st.columns(n_cols)

    for i, (label, value, icon) in enumerate(metrics):
        with cols[i % n_cols]:
            # Format value
            if isinstance(value, float):
                if value < 1:
                    display_val = f"{value:.4f}"
                else:
                    display_val = f"{value:.2f}"
            else:
                display_val = str(value)

            icon_str = icon or "📊"
            st.markdown(
                f"""
            <div style="text-align: center; padding: 0.5rem;">
                <div style="font-size: 1.5rem;">{icon_str}</div>
                <div class="tanml-kpi-label" style="justify-content: center;">{label}</div>
                <div class="tanml-kpi-value">{display_val}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )


def progress_indicator(
    current: int,
    total: int,
    label: str = "Progress",
) -> None:
    """
    Display a progress indicator.

    Args:
        current: Current step
        total: Total steps
        label: Progress label
    """
    progress = current / total if total > 0 else 0
    st.progress(progress, text=f"{label}: {current}/{total}")
