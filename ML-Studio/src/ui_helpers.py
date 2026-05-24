"""Reusable Streamlit UI helpers for consistent dashboard pages."""

from __future__ import annotations

from html import escape
from typing import Any, Iterable

import pandas as pd
import streamlit as st

from src.session_manager import get_engineered_data, get_engineered_data_stage_label


def load_ml_studio_css() -> None:
    """Apply the shared ML Studio dashboard styling."""

    st.markdown(
        """
        <style>
        :root {
            --ml-panel: var(--secondary-background-color, transparent);
            --ml-panel-soft: color-mix(in srgb, var(--text-color, currentColor) 6%, transparent);
            --ml-border: rgba(128, 128, 128, 0.28);
            --ml-border-soft: rgba(128, 128, 128, 0.18);
            --ml-text: var(--text-color, inherit);
            --ml-accent: var(--primary-color, #ff4b4b);
        }

        .block-container {
            padding-top: 1.7rem;
            padding-bottom: 3rem;
            max-width: 1280px;
        }

        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {
            color: var(--ml-text);
            letter-spacing: 0;
        }

        h1, h2, h3 {
            color: var(--ml-text);
            letter-spacing: 0;
        }

        .ml-page-header,
        .ml-hero-panel,
        .ml-card,
        .ml-chat-assistant,
        .ml-chat-user {
            background: var(--ml-panel);
            border: 1px solid var(--ml-border);
            border-radius: 8px;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
        }

        .ml-page-header {
            padding: 1.25rem 1.35rem;
            margin-bottom: 1.15rem;
        }

        .ml-page-eyebrow {
            color: var(--ml-accent);
            font-size: 0.76rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            margin-bottom: 0.25rem;
            text-transform: uppercase;
        }

        .ml-page-title {
            color: var(--ml-text);
            font-size: 2rem;
            font-weight: 750;
            line-height: 1.18;
            margin: 0;
        }

        .ml-page-caption {
            color: var(--ml-text);
            font-size: 0.98rem;
            margin: 0.35rem 0 0;
            opacity: 0.72;
        }

        .ml-hero-panel {
            padding: 1.35rem;
            margin-bottom: 1.1rem;
        }

        .ml-card {
            min-height: 100%;
            padding: 1rem;
        }

        .ml-card-title {
            color: var(--ml-text);
            font-size: 1rem;
            font-weight: 700;
            margin-bottom: 0.3rem;
        }

        .ml-card-copy {
            color: var(--ml-text);
            font-size: 0.92rem;
            line-height: 1.5;
            margin: 0;
            opacity: 0.74;
        }

        .ml-muted {
            color: var(--ml-text);
            opacity: 0.72;
        }

        .ml-pill {
            display: inline-flex;
            align-items: center;
            border-radius: 999px;
            border: 1px solid var(--ml-border);
            background: var(--ml-panel-soft);
            color: var(--ml-text);
            font-size: 0.78rem;
            font-weight: 650;
            padding: 0.2rem 0.55rem;
            margin: 0.1rem 0.15rem 0.1rem 0;
        }

        .ml-pill-good {
            background: color-mix(in srgb, #16a34a 16%, transparent);
            border-color: color-mix(in srgb, #16a34a 35%, var(--ml-border));
            color: var(--ml-text);
        }

        .ml-pill-warn {
            background: color-mix(in srgb, #f59e0b 18%, transparent);
            border-color: color-mix(in srgb, #f59e0b 38%, var(--ml-border));
            color: var(--ml-text);
        }

        .ml-sidebar-brand {
            display: flex;
            gap: 0.65rem;
            align-items: center;
            padding: 0.15rem 0 0.35rem;
        }

        .ml-sidebar-title {
            color: var(--ml-text);
            font-size: 1.2rem;
            font-weight: 800;
            line-height: 1.1;
        }

        .ml-sidebar-subtitle {
            color: var(--ml-text);
            font-size: 0.76rem;
            line-height: 1.2;
            opacity: 0.72;
        }

        .ml-chat-row-user {
            display: flex;
            justify-content: flex-end;
            margin: 0.45rem 0;
        }

        .ml-chat-row-assistant {
            display: flex;
            justify-content: flex-start;
            margin: 0.45rem 0;
        }

        .ml-chat-user,
        .ml-chat-assistant {
            max-width: 78%;
            padding: 0.72rem 0.85rem;
            line-height: 1.45;
        }

        .ml-chat-user {
            background: color-mix(in srgb, var(--ml-accent) 16%, var(--ml-panel));
            border-color: color-mix(in srgb, var(--ml-accent) 36%, var(--ml-border));
        }

        .ml-chat-assistant {
            background: var(--ml-panel);
        }

        div[data-testid="stMetric"] {
            background: var(--ml-panel);
            border: 1px solid var(--ml-border);
            border-radius: 8px;
            padding: 0.85rem 0.95rem;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.035);
        }

        div[data-testid="stMetric"] label {
            color: var(--ml-text);
            opacity: 0.72;
        }

        div[data-testid="stMetricValue"] {
            color: var(--ml-text);
            font-size: 1.2rem;
            font-weight: 750;
            line-height: 1.2;
            overflow-wrap: anywhere;
            white-space: normal;
        }

        div[data-testid="stMetricValue"] > div {
            overflow: visible !important;
            text-overflow: clip !important;
            white-space: normal !important;
        }

        div[data-testid="stDataFrame"] {
            border: 1px solid var(--ml-border-soft);
            border-radius: 8px;
            overflow: hidden;
        }

        .stButton > button {
            border-radius: 7px;
            font-weight: 650;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_page_header(title: str, caption: str, eyebrow: str | None = None) -> None:
    """Render a compact dashboard page header."""

    eyebrow_markup = (
        f"<div class='ml-page-eyebrow'>{escape(eyebrow)}</div>"
        if eyebrow
        else ""
    )
    st.markdown(
        f"""
        <div class="ml-page-header">
            {eyebrow_markup}
            <div class="ml-page-title">{escape(title)}</div>
            <div class="ml-page-caption">{escape(caption)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_dashboard_card(title: str, body: str) -> None:
    """Render a simple fixed-radius informational card."""

    st.markdown(
        f"""
        <div class="ml-card">
            <div class="ml-card-title">{escape(title)}</div>
            <p class="ml-card-copy">{escape(body)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_row(items: Iterable[tuple[str, Any]], columns: int | None = None) -> None:
    """Render metric cards from label/value pairs."""

    metric_items = list(items)
    if not metric_items:
        return

    metric_columns = st.columns(columns or len(metric_items))
    for index, (label, value) in enumerate(metric_items):
        metric_columns[index % len(metric_columns)].metric(label, value)


def render_key_value_table(data: dict[str, Any], *, key_name: str = "Item", value_name: str = "Value") -> None:
    """Display a compact key-value dataframe."""

    st.dataframe(
        pd.DataFrame(
            {
                key_name: list(data.keys()),
                value_name: list(data.values()),
            }
        ),
        width="stretch",
        hide_index=True,
    )


def render_engineered_data_section(
    *,
    session_state,
    preview_rows: int = 8,
    expanded: bool = False,
) -> None:
    """Render a compact preview of the latest transformed dataset."""

    engineered_data = get_engineered_data(session_state)
    if engineered_data is None:
        return

    stage_label = get_engineered_data_stage_label(session_state)
    with st.expander("Active Workflow Dataset", expanded=expanded):
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        metric_col1.metric("Current Stage", stage_label)
        metric_col2.metric("Rows", engineered_data.shape[0])
        metric_col3.metric("Columns", engineered_data.shape[1])
        st.caption("This preview stays synchronized with the latest preprocessing or feature-engineering output.")
        st.dataframe(engineered_data.head(preview_rows), width="stretch")
