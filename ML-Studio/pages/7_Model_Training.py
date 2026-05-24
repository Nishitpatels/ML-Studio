# =========================================================
# FILE: pages/7_Model_Training.py
# =========================================================


import pandas as pd
import streamlit as st

from src.model_selector import perform_model_analysis
from src.session_manager import (
    get_missing_or_empty_keys,
    get_modeling_dataset,
    initialize_session_state,
    set_training_results,
)
from src.ui_helpers import load_ml_studio_css, render_engineered_data_section, render_metric_row, render_page_header
from src.training import generate_training_summary, train_all_models
from src.visualization import plot_metric_bar_chart, plot_model_metric_comparison


st.set_page_config(
    page_title="Model Training | ML Studio",
    page_icon=":material/model_training:",
    layout="wide",
)

load_ml_studio_css()
render_page_header(
    "AutoML Model Training",
    "Train, compare, and select the strongest model on the active workflow dataset.",
    "Model Training",
)
initialize_session_state(st.session_state)

required_session_keys = ["dataset", "target_column", "preprocessing_results"]
missing_keys = get_missing_or_empty_keys(st.session_state, required_session_keys)
if missing_keys:
    st.warning(f"Missing required workflow artifacts: {', '.join(missing_keys)}")
    st.stop()

dataset = get_modeling_dataset(st.session_state)
target_column = st.session_state.target_column
preprocessing_results = st.session_state.preprocessing_results

model_analysis = perform_model_analysis(
    dataframe=dataset,
    target_column=target_column,
)

problem_type = model_analysis["problem_type"]
available_models = model_analysis["available_models"]
recommendations = model_analysis["recommendations"]
model_information = model_analysis["model_information"]


def get_metric_preferences(problem_type: str) -> dict[str, bool]:
    if problem_type == "classification":
        return {
            "Accuracy": True,
            "Precision": True,
            "Recall": True,
            "F1 Score": True,
            "Primary Metric": True,
        }
    return {
        "R2 Score": True,
        "MAE": False,
        "MSE": False,
        "RMSE": False,
        "Primary Metric": True,
    }


def render_training_results(training_results):
    evaluation_dataframe = training_results["evaluation_dataframe"].copy()
    best_model_name = training_results.get("best_model_name")
    best_model_results = training_results.get("best_model")

    if best_model_results is None:
        st.error("No selected model completed successfully. Review the model training issues below.")
        if training_results.get("errors"):
            with st.expander("Model training issues"):
                st.json(training_results["errors"])
        return

    st.markdown("---")
    with st.container(border=True):
        st.subheader("Training Summary")
        summary = generate_training_summary(training_results)
        render_metric_row(
            [
                ("Problem Type", summary["Problem Type"].title()),
                ("Models Trained", summary["Total Models Trained"]),
                ("Best Model", summary["Best Model"]),
            ],
            columns=3,
        )

    st.markdown("---")
    st.subheader("Model Leaderboard")
    st.dataframe(evaluation_dataframe, width="stretch")

    if training_results.get("errors"):
        with st.expander("Model training issues"):
            st.json(training_results["errors"])

    metric_preferences = get_metric_preferences(problem_type)
    available_metrics = [
        metric
        for metric in metric_preferences
        if metric in evaluation_dataframe.columns
    ]

    st.markdown("---")
    st.subheader("Model Performance Comparison")
    selected_metric = st.selectbox(
        "Comparison Metric",
        options=available_metrics,
        index=0,
        key="training_metric_selector",
    )
    comparison_chart = plot_model_metric_comparison(
        evaluation_dataframe,
        selected_metric,
        best_model_name,
        higher_is_better=metric_preferences[selected_metric],
    )
    if comparison_chart is not None:
        st.plotly_chart(comparison_chart, width="stretch")

    st.markdown("---")
    with st.container(border=True):
        st.subheader("Best Fitted Model")
        st.success(f"Best Performing Model: {best_model_name}")

        metrics = best_model_results["metrics"]
        render_metric_row([(metric_name, metric_value) for metric_name, metric_value in metrics.items()])

    metric_chart = plot_metric_bar_chart(
        metrics,
        "Best Model Metrics",
    )
    if metric_chart is not None:
        st.plotly_chart(metric_chart, width="stretch")

    st.markdown("---")
    st.subheader("Best Model Pipeline")
    st.code(str(best_model_results["pipeline"]))


with st.sidebar:
    st.markdown("## Training Controls")
    selected_models = st.multiselect(
        "Select Models to Train",
        options=available_models,
        default=available_models,
    )
    show_model_descriptions = st.checkbox(
        "Show Model Descriptions",
        value=True,
    )

st.markdown("---")
with st.container(border=True):
    st.subheader("Training Dataset Summary")
    render_metric_row(
        [
            ("Rows", dataset.shape[0]),
            ("Columns", dataset.shape[1]),
            ("Problem Type", problem_type.title()),
            ("Models Available", len(available_models)),
        ],
        columns=4,
    )

st.markdown("---")
st.subheader("ML Studio Recommendations")
for recommendation in recommendations:
    st.info(recommendation)

if show_model_descriptions:
    st.markdown("---")
    st.subheader("Model Descriptions")
    for model_name, model_info in model_information.items():
        with st.container(border=True):
            name_col, description_col, action_col = st.columns([1.6, 3.4, 1.1], gap="large")
            name_col.markdown(f"**{model_name}**")
            description_col.write(model_info["description"])
            with action_col.popover("Overview"):
                st.write(model_info["when_to_use"])
                st.caption("Strengths")
                for strength in model_info["strengths"]:
                    st.write(f"- {strength}")
                st.caption("Weaknesses")
                for weakness in model_info["watch_outs"]:
                    st.write(f"- {weakness}")

st.markdown("---")
train_models_button = st.button("Train Selected Models")

results_to_display = None
if train_models_button:
    if len(selected_models) == 0:
        st.warning("Please select at least one model.")
    else:
        with st.spinner("Training ML models..."):
            try:
                training_results = train_all_models(
                    dataframe=dataset,
                    target_column=target_column,
                    preprocessing_results=preprocessing_results,
                    selected_models=selected_models,
                )
                set_training_results(st.session_state, training_results)
                st.success("Model training completed successfully.")
                results_to_display = training_results
            except Exception as error:
                st.error(f"Training failed: {error}")

if results_to_display is None and st.session_state.training_completed and st.session_state.training_results is not None:
    st.markdown("---")
    st.success("Models are already trained in the current session.")
    results_to_display = st.session_state.training_results

if results_to_display is not None:
    render_training_results(results_to_display)
    render_engineered_data_section(session_state=st.session_state)
    st.markdown("---")
    st.info("Next Step -> Navigate to 'Explainability' page.")
