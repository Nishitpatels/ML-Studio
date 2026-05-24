# =========================================================
# FILE: pages/16_Experiment_Tracking.py
# =========================================================


import pandas as pd
import streamlit as st

from src.experiment_tracker import (
    experiments_to_dataframe,
    get_best_experiment,
    load_all_experiments,
    save_experiment,
)
from src.session_manager import get_missing_or_empty_keys, get_modeling_dataset, initialize_session_state
from src.ui_helpers import load_ml_studio_css, render_engineered_data_section, render_metric_row, render_page_header


st.set_page_config(
    page_title="Experiment Tracking | ML Studio",
    page_icon=":material/science:",
    layout="wide"
)

load_ml_studio_css()
render_page_header(
    "Experiment Tracking",
    "Track, save, and compare model experiments from the current workflow.",
    "Experiments",
)

initialize_session_state(st.session_state)
required_keys = [
    "best_model",
    "training_results",
    "dataset"
]

missing_keys = get_missing_or_empty_keys(st.session_state, required_keys)
if missing_keys:
    st.warning(f"Missing required workflow artifacts: {', '.join(missing_keys)}")
    st.stop()

best_model_results = st.session_state.best_model
training_results = st.session_state.training_results
dataset = get_modeling_dataset(st.session_state)

model_name = best_model_results["model_name"]
metrics = best_model_results["metrics"]
pipeline = best_model_results["pipeline"]
problem_type = training_results["problem_type"]

with st.sidebar:
    st.markdown("## Experiment Controls")
    experiment_name = st.text_input(
        "Experiment Alias",
        value=model_name.replace(" ", "_")
    )

st.markdown("---")
with st.container(border=True):
    st.subheader("Current Experiment")
    render_metric_row(
        [
            ("Model", model_name),
            ("Problem Type", problem_type.title()),
            ("Dataset Rows", dataset.shape[0]),
        ],
        columns=3,
    )

st.markdown("---")
with st.container(border=True):
    st.subheader("Current Metrics")
    render_metric_row([(metric_name, metric_value) for metric_name, metric_value in metrics.items()])

st.markdown("---")
save_experiment_button = st.button("Save Experiment")

if save_experiment_button:
    try:
        with st.spinner("Saving experiment..."):
            parameters = pipeline.get_params() if hasattr(pipeline, "get_params") else {}

            experiment_data = save_experiment(
                model_name=experiment_name,
                metrics=metrics,
                parameters=parameters,
                problem_type=problem_type,
                dataset_shape=dataset.shape
            )

            st.success("Experiment saved successfully.")

            st.markdown("---")
            st.subheader("Saved Experiment")

            with st.container(border=True):
                preview_col1, preview_col2 = st.columns(2)
                preview_col1.metric("Experiment ID", experiment_data["Experiment ID"])
                preview_col2.metric(
                    "Tracked Parameters",
                    len(experiment_data.get("Parameters", {})),
                )

            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Timestamp": experiment_data["Timestamp"],
                            "Model": experiment_data["Model"],
                            "Problem Type": experiment_data["Problem Type"],
                            "Dataset Rows": experiment_data["Dataset Rows"],
                            "Dataset Columns": experiment_data["Dataset Columns"],
                        }
                    ]
                ),
                width="stretch",
            )

            with st.expander("View Saved Parameters"):
                st.json(experiment_data.get("Parameters", {}))

            with st.expander("View Saved Metrics"):
                st.json(experiment_data.get("Metrics", {}))

    except Exception as error:
        st.error(f"Experiment saving failed: {error}")

st.markdown("---")
st.subheader("Experiment History")

try:
    experiments = load_all_experiments()

    if len(experiments) == 0:
        st.info("No experiments found yet.")
    else:
        experiments_dataframe = experiments_to_dataframe(experiments)
        st.dataframe(experiments_dataframe, width="stretch")

        st.markdown("---")
        st.subheader("Parameter Details")
        for experiment in sorted(experiments, key=lambda item: item.get("Timestamp", ""), reverse=True):
            expander_label = (
                f"{experiment['Experiment ID']} | {experiment['Model']} | {experiment['Timestamp']}"
            )
            with st.expander(expander_label):
                summary_col1, summary_col2, summary_col3 = st.columns(3)
                summary_col1.metric("Problem Type", experiment["Problem Type"].title())
                summary_col2.metric("Rows", experiment["Dataset Rows"])
                summary_col3.metric("Parameters", len(experiment.get("Parameters", {})))
                st.caption("Metrics")
                st.json(experiment.get("Metrics", {}))
                st.caption("Parameters")
                st.json(experiment.get("Parameters", {}))

        st.markdown("---")
        st.subheader("Best Experiment")
        best_experiment = get_best_experiment(experiments_dataframe)
        if best_experiment is not None:
            st.success(f"Best Model: {best_experiment['Model']}")
            st.dataframe(pd.DataFrame([best_experiment]), width="stretch")

except Exception as error:
    st.error(f"Experiment loading failed: {error}")

render_engineered_data_section(session_state=st.session_state)
st.markdown("---")
st.success("Experiment tracking system ready.")
