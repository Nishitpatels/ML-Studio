from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.session_manager import (
    get_modeling_dataset,
    initialize_session_state as initialize_shared_session_state,
)
from src.settings_manager import generate_storage_summary, initialize_directories
from src.ui_helpers import (
    load_ml_studio_css,
    render_dashboard_card,
    render_metric_row,
)


APP_ROOT = Path(__file__).parent
LOGO_PATH = APP_ROOT / "src" / "images" / "logo.png"


st.set_page_config(
    page_title="ML Studio",
    page_icon=":material/analytics:",
    layout="wide",
    initial_sidebar_state="expanded",
)


def initialize_session_state() -> None:
    """Initialize the shared ML Studio session state."""

    initialize_shared_session_state(st.session_state)
    initialize_directories()


def reset_session_state() -> None:
    """Clear project state and let the shared initializer rebuild defaults."""

    for key in list(st.session_state.keys()):
        del st.session_state[key]


def _workflow_steps() -> list[tuple[str, bool]]:
    return [
        ("Upload Dataset", st.session_state.dataset is not None),
        ("Dataset Overview", st.session_state.eda_completed),
        ("Preprocessing", st.session_state.preprocessing_completed),
        ("Feature Engineering", st.session_state.feature_engineering_completed),
        ("Model Training", st.session_state.training_completed),
    ]


def render_sidebar() -> None:
    """Render the application sidebar with compact workflow context."""

    with st.sidebar:
        brand_col, text_col = st.columns([0.35, 1.0], vertical_alignment="center")
        if LOGO_PATH.exists():
            brand_col.image(str(LOGO_PATH), width=46)
        text_col.markdown(
            """
            <div class="ml-sidebar-title">ML Studio</div>
            <div class="ml-sidebar-subtitle">AI-Assisted Machine Learning Studio</div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()

        active_dataset = get_modeling_dataset(st.session_state)
        st.markdown("### Dataset")
        if st.session_state.dataset is not None and active_dataset is not None:
            st.success("Dataset loaded")
            st.caption(st.session_state.current_dataset_name or "Unnamed dataset")
            st.metric("Rows", active_dataset.shape[0])
            st.metric("Columns", active_dataset.shape[1])
        else:
            st.info("Upload a dataset to begin.")

        st.divider()

        st.markdown("### Workflow")
        steps = _workflow_steps()
        completed_steps = sum(1 for _, complete in steps if complete)
        st.progress(completed_steps / len(steps), text=f"{completed_steps} of {len(steps)} steps complete")
        for step, complete in steps:
            status = "Complete" if complete else "Pending"
            st.caption(f"{step}: {status}")

        st.divider()

        if st.button("Reset Project State", width="stretch"):
            reset_session_state()
            st.rerun()

        st.divider()
        st.markdown("### Workspace")
        st.caption("A focused ML workflow for preprocessing, training, explainability, reports, and export.")


def render_hero_header() -> None:
    """Render the ML Studio dashboard header."""

    with st.container(border=True):
        logo_col, title_col, status_col = st.columns([0.55, 3.2, 1.35], vertical_alignment="center")
        if LOGO_PATH.exists():
            logo_col.image(str(LOGO_PATH), width=74)
        else:
            logo_col.markdown("### ML")

        title_col.markdown(
            """
            <div class="ml-page-eyebrow">Machine Learning Workspace</div>
            <div class="ml-page-title">ML Studio</div>
            <div class="ml-page-caption">AI-Assisted Machine Learning Studio</div>
            """,
            unsafe_allow_html=True,
        )

        active_dataset = get_modeling_dataset(st.session_state)
        if active_dataset is not None:
            status_col.metric("Active Dataset", st.session_state.current_dataset_name or "Loaded")
        else:
            status_col.metric("Active Dataset", "Not loaded")


def render_quick_stats() -> None:
    dataset = get_modeling_dataset(st.session_state)
    best_model = st.session_state.get("best_model")
    target_column = st.session_state.get("target_column") or "Not selected"
    problem_type = st.session_state.get("problem_type") or "Not detected"

    render_metric_row(
        [
            ("Rows", dataset.shape[0] if dataset is not None else 0),
            ("Columns", dataset.shape[1] if dataset is not None else 0),
            ("Target", target_column),
            ("Problem Type", str(problem_type).title()),
            ("Best Model", best_model.get("model_name") if best_model else "Not trained"),
        ],
        columns=5,
    )


def render_workflow_status() -> None:
    st.subheader("Workflow Status")
    columns = st.columns(len(_workflow_steps()))
    for index, (step, complete) in enumerate(_workflow_steps()):
        with columns[index]:
            with st.container(border=True):
                st.markdown(f"**{step}**")
                if complete:
                    st.success("Complete")
                else:
                    st.info("Pending")


def render_feature_overview() -> None:
    st.subheader("Feature Overview")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_dashboard_card(
            "Dataset Intelligence",
            "Health checks, missing values, skewness, outliers, and quality recommendations.",
        )
    with col2:
        render_dashboard_card(
            "Preprocessing",
            "Manual, transparent controls for imputation, encoding, scaling, duplicates, and splits.",
        )
    with col3:
        render_dashboard_card(
            "Modeling",
            "Train, compare, tune, evaluate, and export models from the same workflow state.",
        )
    with col4:
        render_dashboard_card(
            "AI Assistance",
            "Dataset chat, ML recommendations, explainability summaries, and report generation.",
        )


def render_recent_dataset_summary() -> None:
    st.subheader("Recent Dataset Summary")
    dataset = get_modeling_dataset(st.session_state)
    if dataset is None:
        with st.container(border=True):
            st.info("No dataset is active yet. Use the Upload Dataset page to start a workflow.")
        return

    summary_frame = pd.DataFrame(
        {
            "Column": dataset.columns,
            "Data Type": dataset.dtypes.astype(str),
            "Missing Values": dataset.isna().sum().values,
            "Unique Values": dataset.nunique(dropna=True).values,
        }
    )
    with st.container(border=True):
        render_metric_row(
            [
                ("Missing Values", int(dataset.isna().sum().sum())),
                ("Duplicate Rows", int(dataset.duplicated().sum())),
                ("Memory MB", round(dataset.memory_usage(deep=True).sum() / (1024 * 1024), 2)),
            ],
            columns=3,
        )
        st.dataframe(summary_frame.head(12), width="stretch", hide_index=True)


def render_recent_model_performance() -> None:
    st.subheader("Recent Model Performance")
    best_model = st.session_state.get("best_model")
    training_results = st.session_state.get("training_results")
    if not best_model:
        with st.container(border=True):
            st.info("No trained model is available yet. Run Model Training to populate this panel.")
        return

    metrics = best_model.get("metrics", {})
    with st.container(border=True):
        render_metric_row([(name, value) for name, value in metrics.items()], columns=max(len(metrics), 1))
        leaderboard = training_results.get("evaluation_dataframe") if training_results else None
        if leaderboard is not None:
            st.caption("Latest leaderboard")
            st.dataframe(leaderboard, width="stretch")


def render_workflow_diagram() -> None:
    st.subheader("Workflow Diagram")
    st.graphviz_chart(
        """
        digraph {
            graph [rankdir=LR, bgcolor="transparent", pad="0.2", nodesep="0.45", ranksep="0.45"];
            node [shape=box, style="rounded,filled", fillcolor="#ffffff", color="#d9e2ec", fontname="Arial", fontsize=10];
            edge [color="#64748b", arrowsize=0.7];
            Upload -> Overview -> EDA -> Preprocessing -> "Feature Engineering" -> Training -> Evaluation -> Export;
            Training -> Tuning;
            Evaluation -> Explainability;
            Evaluation -> Reports;
        }
        """,
        use_container_width=True,
    )


def render_system_status() -> None:
    st.subheader("System Status")
    storage_summary = generate_storage_summary()
    session_keys = len(st.session_state.keys())
    dataset_ready = "Ready" if st.session_state.dataset is not None else "Waiting"
    model_ready = "Ready" if st.session_state.best_model is not None else "Waiting"

    render_metric_row(
        [
            ("Dataset State", dataset_ready),
            ("Model State", model_ready),
            ("Session Keys", session_keys),
            ("Reports", storage_summary.get("Reports", 0)),
            ("Experiments", storage_summary.get("Experiments", 0)),
        ],
        columns=5,
    )


def main() -> None:
    load_ml_studio_css()
    initialize_session_state()
    render_sidebar()

    render_hero_header()

    st.markdown("### Quick Stats")
    render_quick_stats()

    st.markdown("---")
    render_workflow_status()

    st.markdown("---")
    render_feature_overview()

    st.markdown("---")
    dataset_col, model_col = st.columns(2, gap="large")
    with dataset_col:
        render_recent_dataset_summary()
    with model_col:
        render_recent_model_performance()

    st.markdown("---")
    diagram_col, system_col = st.columns([1.35, 1.0], gap="large")
    with diagram_col:
        render_workflow_diagram()
    with system_col:
        render_system_status()


if __name__ == "__main__":
    main()
