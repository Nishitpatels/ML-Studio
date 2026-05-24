# =========================================================
# FILE: pages/4_EDA.py
# =========================================================


import pandas as pd
import streamlit as st

from src.analyzer import detect_column_types
from src.session_manager import get_modeling_dataset, initialize_session_state
from src.ui_helpers import load_ml_studio_css, render_engineered_data_section, render_page_header
from src.visualization import (
    plot_boxplot,
    plot_categorical_distribution,
    plot_class_balance,
    plot_column_type_distribution,
    plot_correlation_heatmap,
    plot_feature_distribution,
    plot_kde_distribution,
    plot_missing_values,
    plot_numerical_summary,
    plot_outlier_analysis,
    plot_scatter,
    plot_target_distribution,
)


st.set_page_config(
    page_title="EDA Dashboard | ML Studio",
    page_icon=":material/analytics:",
    layout="wide",
)

load_ml_studio_css()
render_page_header(
    "Exploratory Data Analysis",
    "Inspect distributions, missingness, correlations, targets, scatter plots, and outliers.",
    "EDA Dashboard",
)
initialize_session_state(st.session_state)

dataset = get_modeling_dataset(st.session_state)
target_column = st.session_state.target_column

if dataset is None:
    st.warning("No dataset found. Please upload a dataset first.")
    st.stop()

column_types = detect_column_types(dataset)
numerical_columns = column_types["numerical_columns"]
categorical_columns = column_types["categorical_columns"]
datetime_columns = column_types["datetime_columns"]

with st.sidebar:
    st.markdown("## EDA Controls")
    selected_analysis = st.radio(
        "Select Analysis Type",
        options=[
            "Dataset Overview",
            "Missing Values",
            "Correlation Analysis",
            "Numerical Analysis",
            "Categorical Analysis",
            "Target Analysis",
            "Scatter Plot Analysis",
            "Outlier Analysis",
        ],
    )

with st.container(border=True):
    stage_col1, stage_col2, stage_col3 = st.columns(3)
    stage_col1.metric("Rows", dataset.shape[0])
    stage_col2.metric("Columns", dataset.shape[1])
    stage_col3.metric(
        "Dataset Stage",
        "Feature Engineered"
        if st.session_state.feature_engineering_completed
        else "Processed"
        if st.session_state.preprocessing_completed
        else "Uploaded",
    )

if selected_analysis == "Dataset Overview":
    st.subheader("Dataset Structure Overview")

    overview_col1, overview_col2, overview_col3 = st.columns(3)
    overview_col1.metric("Numerical Features", len(numerical_columns))
    overview_col2.metric("Categorical Features", len(categorical_columns))
    overview_col3.metric("Datetime Features", len(datetime_columns))

    figure = plot_column_type_distribution(
        numerical_count=len(numerical_columns),
        categorical_count=len(categorical_columns),
        datetime_count=len(datetime_columns),
    )
    st.plotly_chart(figure, width="stretch")

    st.markdown("---")
    st.subheader("Numerical Feature Summary")
    numerical_summary_chart = plot_numerical_summary(dataset)
    if numerical_summary_chart is not None:
        st.plotly_chart(numerical_summary_chart, width="stretch")
    else:
        st.info("No numerical columns available.")

elif selected_analysis == "Missing Values":
    st.subheader("Missing Value Visualization")
    figure = plot_missing_values(dataset)
    if figure is not None:
        st.plotly_chart(figure, width="stretch")
    else:
        st.success("No missing values detected.")

elif selected_analysis == "Correlation Analysis":
    st.subheader("Correlation Heatmap")
    figure = plot_correlation_heatmap(dataset)
    if figure is not None:
        st.plotly_chart(figure, width="stretch")
    else:
        st.info("At least two numerical columns are required.")

elif selected_analysis == "Numerical Analysis":
    st.subheader("Numerical Feature Analysis")
    if len(numerical_columns) == 0:
        st.warning("No numerical columns detected.")
    else:
        selected_numerical_column = st.selectbox(
            "Select Numerical Feature",
            options=numerical_columns,
        )

        distribution_chart = plot_feature_distribution(
            dataset,
            selected_numerical_column,
        )
        st.plotly_chart(distribution_chart, width="stretch")

        boxplot_chart = plot_boxplot(
            dataset,
            selected_numerical_column,
        )
        st.plotly_chart(boxplot_chart, width="stretch")

        st.markdown("---")
        st.subheader("Probability Density (KDE / PDF)")
        kde_columns = st.multiselect(
            "Select Numerical Columns",
            options=numerical_columns,
            default=[selected_numerical_column],
            help="Overlay density curves to compare skewness and distribution shape.",
        )
        kde_chart = plot_kde_distribution(dataset, kde_columns)
        if kde_chart is not None:
            st.plotly_chart(kde_chart, width="stretch")
        else:
            st.info("Select columns with at least two distinct numeric values to generate a KDE plot.")

        st.markdown("---")
        st.subheader("Statistical Summary")
        statistics = dataset[selected_numerical_column].describe().rename("Value")
        st.dataframe(statistics.to_frame(), width="stretch")

elif selected_analysis == "Categorical Analysis":
    st.subheader("Categorical Feature Analysis")
    if len(categorical_columns) == 0:
        st.warning("No categorical columns detected.")
    else:
        selected_categorical_column = st.selectbox(
            "Select Categorical Feature",
            options=categorical_columns,
        )
        categorical_chart = plot_categorical_distribution(
            dataset,
            selected_categorical_column,
        )
        st.plotly_chart(categorical_chart, width="stretch")

        st.markdown("---")
        st.subheader("Category Counts")
        value_counts = dataset[selected_categorical_column].value_counts(dropna=False)
        value_counts_dataframe = pd.DataFrame(
            {
                "Category": value_counts.index.astype(str),
                "Count": value_counts.values,
            }
        )
        st.dataframe(value_counts_dataframe, width="stretch")

elif selected_analysis == "Target Analysis":
    st.subheader("Target Analysis")
    if target_column is None or target_column not in dataset.columns:
        st.warning("No valid target column is available in the current dataset.")
    else:
        target_chart = plot_target_distribution(dataset, target_column)
        st.plotly_chart(target_chart, width="stretch")

        if dataset[target_column].nunique(dropna=True) <= 20:
            st.markdown("---")
            st.subheader("Class Balance")
            class_balance_chart = plot_class_balance(dataset, target_column)
            st.plotly_chart(class_balance_chart, width="stretch")

elif selected_analysis == "Scatter Plot Analysis":
    st.subheader("Scatter Plot Analysis")
    if len(numerical_columns) < 2:
        st.warning("At least two numerical columns are required.")
    else:
        scatter_col1, scatter_col2 = st.columns(2)
        with scatter_col1:
            x_column = st.selectbox("Select X-Axis", options=numerical_columns)
        with scatter_col2:
            y_column = st.selectbox(
                "Select Y-Axis",
                options=numerical_columns,
                index=1,
            )
        color_options = [None] + categorical_columns
        color_column = st.selectbox("Color By (Optional)", options=color_options)
        scatter_chart = plot_scatter(
            dataset,
            x_column=x_column,
            y_column=y_column,
            color_column=color_column,
        )
        st.plotly_chart(scatter_chart, width="stretch")

elif selected_analysis == "Outlier Analysis":
    st.subheader("Outlier Visualization")
    if len(numerical_columns) == 0:
        st.warning("No numerical columns available.")
    else:
        selected_column = st.selectbox(
            "Select Numerical Column",
            options=numerical_columns,
        )
        outlier_chart = plot_outlier_analysis(dataset, selected_column)
        st.plotly_chart(outlier_chart, width="stretch")

st.session_state.eda_completed = True

render_engineered_data_section(session_state=st.session_state)
st.markdown("---")
st.success("EDA completed successfully.")
st.info("Next Step -> Navigate to 'Preprocessing' page.")
