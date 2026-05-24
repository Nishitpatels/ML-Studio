import pandas as pd
import streamlit as st

from src.hyperparameter_tuning import (
    build_tuned_model_results,
    generate_tuning_explanation,
    get_parameter_columns,
    merge_tuned_model_with_training_results,
    tune_model,
)
from src.model_selector import detect_problem_type, get_models_by_problem_type
from src.session_manager import (
    get_missing_or_empty_keys,
    get_modeling_dataset,
    initialize_session_state,
    set_tuning_results,
)
from src.ui_helpers import load_ml_studio_css, render_engineered_data_section, render_metric_row, render_page_header
from src.visualization import plot_parameter_impact, plot_tuning_progression


st.set_page_config(
    page_title="Hyperparameter Tuning | ML Studio",
    page_icon=":material/tune:",
    layout="wide",
)

load_ml_studio_css()
render_page_header(
    "Hyperparameter Tuning",
    "Optimize a model, inspect score changes, and understand why the best parameters won.",
    "Model Optimization",
)
initialize_session_state(st.session_state)

required_keys = ["dataset", "target_column", "preprocessing_results"]
missing_keys = get_missing_or_empty_keys(st.session_state, required_keys)
if missing_keys:
    st.warning(f"Missing required workflow artifacts: {', '.join(missing_keys)}")
    st.stop()

dataset = get_modeling_dataset(st.session_state)
target_column = st.session_state.target_column
preprocessing_results = st.session_state.preprocessing_results

problem_type = detect_problem_type(dataset[target_column])
models = get_models_by_problem_type(problem_type)

with st.sidebar:
    st.markdown("## Tuning Controls")
    selected_model_name = st.selectbox("Select Model", options=list(models.keys()))
    tuning_method = st.radio("Tuning Method", options=["grid", "random"])

st.markdown("---")
with st.container(border=True):
    st.subheader("Tuning Configuration")
    render_metric_row(
        [
            ("Problem Type", problem_type.title()),
            ("Selected Model", selected_model_name),
            ("Tuning Method", tuning_method.title()),
        ],
        columns=3,
    )

st.markdown("---")
start_tuning = st.button("Start Hyperparameter Tuning")

results_to_display = st.session_state.get("tuning_results")

if start_tuning:
    try:
        with st.spinner("Optimizing hyperparameters..."):
            selected_model = models[selected_model_name]
            tuning_results = tune_model(
                model_name=selected_model_name,
                model=selected_model,
                preprocessor=preprocessing_results.get(
                    "fitted_preprocessing_transformer",
                    preprocessing_results["preprocessor"],
                ),
                X_train=preprocessing_results.get("X_train_raw", preprocessing_results["X_train"]),
                y_train=preprocessing_results["y_train"],
                tuning_method=tuning_method,
            )

            if tuning_results is None:
                st.warning("Tuning is not supported for the selected model.")
            else:
                tuned_model_results = build_tuned_model_results(
                    selected_model_name,
                    tuning_results,
                    preprocessing_results.get("X_test_raw", preprocessing_results["X_test"]),
                    preprocessing_results["y_test"],
                    problem_type,
                )
                training_results = merge_tuned_model_with_training_results(
                    st.session_state.get("training_results"),
                    tuned_model_results,
                    problem_type,
                )
                set_tuning_results(
                    st.session_state,
                    tuning_results,
                    training_results,
                )
                results_to_display = tuning_results
                st.success("Hyperparameter tuning completed. The tuned model is now active downstream.")
    except Exception as error:
        st.error(f"Tuning failed: {error}")

if results_to_display is not None:
    st.markdown("---")
    with st.container(border=True):
        st.subheader("Best Tuning Results")
        render_metric_row(
            [
                ("Best CV Score", round(results_to_display["best_score"], 4)),
                ("Model", results_to_display["model_name"]),
                ("Scoring Metric", results_to_display.get("scoring_metric", "accuracy")),
            ],
            columns=3,
        )

    st.markdown("---")
    st.subheader("Best Parameters")
    parameter_dataframe = pd.DataFrame(
        {
            "Parameter": results_to_display["best_parameters"].keys(),
            "Value": results_to_display["best_parameters"].values(),
        }
    )
    st.dataframe(parameter_dataframe, width="stretch", hide_index=True)

    st.markdown("---")
    st.subheader("Why These Parameters Were Selected")
    for explanation in generate_tuning_explanation(results_to_display):
        st.info(explanation)

    st.markdown("---")
    st.subheader("Tuning Progression")
    progression_chart = plot_tuning_progression(results_to_display["cv_results"])
    if progression_chart is not None:
        st.plotly_chart(progression_chart, width="stretch")

    st.markdown("---")
    st.subheader("Parameter Impact")
    parameter_columns = get_parameter_columns(results_to_display["cv_results"])
    if parameter_columns:
        selected_parameter = st.selectbox(
            "Compare Parameter vs Score",
            options=parameter_columns,
            format_func=lambda value: value.replace("param_model__", "").replace("param_", ""),
        )
        parameter_chart = plot_parameter_impact(results_to_display["cv_results"], selected_parameter)
        if parameter_chart is not None:
            st.plotly_chart(parameter_chart, width="stretch")
    else:
        st.info("No tunable parameter columns were available for visualization.")

    st.markdown("---")
    st.subheader("Score Comparison Table")
    cv_results = results_to_display["cv_results"].copy()
    comparison_columns = [
        column
        for column in cv_results.columns
        if column in {"candidate_id", "mean_test_score", "std_test_score", "rank_test_score"}
        or column.startswith("param_")
    ]
    st.dataframe(
        cv_results[comparison_columns].sort_values("rank_test_score"),
        width="stretch",
        hide_index=True,
    )

    st.markdown("---")
    st.subheader("Best Estimator")
    st.code(str(results_to_display["best_estimator"]))

    tuned_model = st.session_state.get("training_results", {}).get("best_model")
    if tuned_model is not None:
        st.markdown("---")
        with st.container(border=True):
            st.subheader("Tuned Model Test Metrics")
            render_metric_row([(metric_name, metric_value) for metric_name, metric_value in tuned_model["metrics"].items()])

render_engineered_data_section(session_state=st.session_state)
