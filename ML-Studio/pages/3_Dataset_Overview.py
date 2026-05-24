import pandas as pd
import streamlit as st

from src.analyzer import perform_complete_analysis
from src.session_manager import get_modeling_dataset, initialize_session_state
from src.ui_helpers import load_ml_studio_css, render_engineered_data_section, render_metric_row, render_page_header
from src.visualization import plot_distribution_with_kde


st.set_page_config(page_title="Dataset Overview | ML Studio", page_icon=":material/monitoring:", layout="wide")
load_ml_studio_css()
initialize_session_state(st.session_state)

render_page_header(
    "Dataset Overview",
    "Review dataset quality, structure, warnings, skewness, and ML readiness recommendations.",
    "Dataset Intelligence",
)

dataset = get_modeling_dataset(st.session_state)
target_column = st.session_state.target_column

if dataset is None:
    st.warning("No dataset found. Please upload a dataset first.")
    st.stop()


@st.cache_data(show_spinner=False)
def _cached_analysis(dataframe, target):
    return perform_complete_analysis(dataframe, target)


analysis_results = _cached_analysis(dataset, target_column)

health_score = analysis_results["health_score"]["health_score"]
warnings = analysis_results["health_score"]["warnings"]
health_col, warning_col = st.columns([1.0, 1.4], gap="large")

with health_col:
    with st.container(border=True):
        st.subheader("Dataset Health Score")
        st.metric("Overall Quality", f"{health_score}/100")
        if health_score >= 85:
            st.success("Strong dataset health")
        elif health_score >= 60:
            st.warning("Dataset needs review")
        else:
            st.error("Dataset needs attention")
        st.caption("Based on missing values, duplicate rows, and cardinality warnings.")

with warning_col:
    with st.container(border=True):
        st.subheader("Warnings")
        if warnings:
            for warning in warnings:
                st.warning(warning)
        else:
            st.success("No major dataset issues detected.")

basic_info = analysis_results["basic_info"]
st.markdown("---")
with st.container(border=True):
    st.subheader("Basic Dataset Information")
    render_metric_row(
        [
            ("Rows", basic_info["rows"]),
            ("Columns", basic_info["columns"]),
            ("Missing Values", basic_info["total_missing_values"]),
            ("Duplicate Rows", basic_info["duplicate_rows"]),
            ("Memory Usage (MB)", basic_info["memory_usage_mb"]),
        ],
        columns=5,
    )

st.markdown("---")
with st.container(border=True):
    st.subheader("Column Type Detection")
    type_col1, type_col2, type_col3 = st.columns(3)
    type_groups = [
        ("Numerical Columns", analysis_results["column_types"]["numerical_columns"]),
        ("Categorical Columns", analysis_results["column_types"]["categorical_columns"]),
        ("Datetime Columns", analysis_results["column_types"]["datetime_columns"]),
    ]
    for column, (title, columns) in zip([type_col1, type_col2, type_col3], type_groups):
        with column:
            st.metric(title, len(columns))
            if columns:
                st.dataframe(pd.DataFrame({"Column": columns}), width="stretch", hide_index=True)
            else:
                st.info("None detected")

if target_column is not None and "target_analysis" in analysis_results:
    st.markdown("---")
    with st.container(border=True):
        st.subheader("Target Column Analysis")
        target_analysis = analysis_results["target_analysis"]
        render_metric_row(
            [
                ("Target Column", target_analysis["target_column"]),
                ("Problem Type", target_analysis["problem_type"]),
                ("Unique Values", target_analysis["unique_values"]),
                ("Missing Values", target_analysis["missing_values"]),
            ],
            columns=4,
        )

st.markdown("---")
missing_col, duplicate_col = st.columns(2, gap="large")
with missing_col:
    with st.container(border=True):
        st.subheader("Missing Value Analysis")
        missing_analysis = analysis_results["missing_analysis"]
        missing_filtered = missing_analysis[missing_analysis["Missing Values"] > 0]
        if not missing_filtered.empty:
            st.dataframe(missing_filtered, width="stretch", hide_index=True)
        else:
            st.success("No missing values detected.")

with duplicate_col:
    with st.container(border=True):
        st.subheader("Duplicate Analysis")
        duplicate_analysis = analysis_results["duplicate_analysis"]
        duplicate_recommendation = analysis_results["duplicate_recommendation"]
        render_metric_row(
            [
                ("Duplicate Rows", duplicate_analysis["duplicate_count"]),
                ("Duplicate Percentage", f"{duplicate_analysis['duplicate_percentage']}%"),
            ],
            columns=2,
        )
        recommendation_type = duplicate_recommendation["action"]
        recommendation_text = f"{duplicate_recommendation['headline']} {duplicate_recommendation['reasoning']}"
        if recommendation_type == "remove":
            st.warning(recommendation_text)
        elif recommendation_type == "review":
            st.info(recommendation_text)
        else:
            st.success(recommendation_text)

st.markdown("---")
with st.container(border=True):
    st.subheader("Cardinality Analysis")
    st.dataframe(analysis_results["cardinality_analysis"], width="stretch", hide_index=True)

st.markdown("---")
with st.container(border=True):
    st.subheader("Skewness Analysis")
    skewness_analysis = analysis_results["skewness_analysis"]
    if not skewness_analysis.empty:
        table_col, action_col = st.columns([3.2, 1.0], gap="large")
        with table_col:
            st.dataframe(skewness_analysis, width="stretch", hide_index=True)
        with action_col:
            st.caption("Distribution Plot")
            selected_skewness_column = st.selectbox(
                "Column",
                options=skewness_analysis["Column"].tolist(),
                key="selected_skewness_column_select",
            )
            if st.button("Show Distribution", width="stretch"):
                st.session_state["selected_skewness_column"] = selected_skewness_column

        selected_skewness_column = st.session_state.get("selected_skewness_column")
        if selected_skewness_column in skewness_analysis["Column"].tolist():
            st.markdown("#### Distribution View")
            distribution_chart = plot_distribution_with_kde(dataset, selected_skewness_column)
            if distribution_chart is not None:
                st.plotly_chart(distribution_chart, width="stretch")
            else:
                st.info("Not enough numeric variation is available to draw this distribution plot.")
    else:
        st.info("No numerical columns available.")

st.markdown("---")
outlier_col, correlation_col = st.columns(2, gap="large")
with outlier_col:
    with st.container(border=True):
        st.subheader("Outlier Analysis")
        outlier_analysis = analysis_results["outlier_analysis"]
        if not outlier_analysis.empty:
            st.dataframe(outlier_analysis, width="stretch", hide_index=True)
        else:
            st.info("No numerical columns available.")

with correlation_col:
    with st.container(border=True):
        st.subheader("High Correlation Detection")
        high_correlations = analysis_results["high_correlations"]
        if not high_correlations.empty:
            st.dataframe(high_correlations, width="stretch", hide_index=True)
        else:
            st.success("No highly correlated features detected.")

if "class_balance" in analysis_results:
    st.markdown("---")
    with st.container(border=True):
        st.subheader("Class Balance Analysis")
        st.dataframe(analysis_results["class_balance"], width="stretch", hide_index=True)

st.markdown("---")
with st.container(border=True):
    st.subheader("ML Studio Recommendations")
    recommendations = analysis_results["recommendations"]
    if recommendations:
        for recommendation in recommendations:
            st.info(recommendation)
    else:
        st.success("Dataset looks healthy for machine learning workflows.")

st.markdown("---")
with st.expander("Active Dataset Inspection"):
    preview_rows = st.slider("Preview Rows", min_value=5, max_value=100, value=20)
    st.dataframe(dataset.head(preview_rows), width="stretch")

render_engineered_data_section(session_state=st.session_state)
st.session_state.eda_completed = True
st.markdown("---")
st.success("Dataset overview completed successfully.")
