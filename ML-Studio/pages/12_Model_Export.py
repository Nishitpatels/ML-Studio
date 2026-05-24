# =========================================================
# FILE: pages/12_Model_Export.py
# =========================================================


import os

import streamlit as st

from src.model_export import (
    export_model_pipeline,
    generate_export_summary,
    generate_model_metadata,
    save_model_metadata,
)
from src.prediction import get_expected_features
from src.session_manager import get_missing_or_empty_keys, get_modeling_dataset, initialize_session_state
from src.ui_helpers import load_ml_studio_css, render_engineered_data_section, render_metric_row, render_page_header


st.set_page_config(
    page_title="Model Export | ML Studio",
    page_icon=":material/package_2:",
    layout="wide"
)

load_ml_studio_css()
render_page_header(
    "Model Export Center",
    "Export trained ML pipelines, metadata, and model metrics from the active workflow.",
    "Export",
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

pipeline = best_model_results["pipeline"]
metrics = best_model_results["metrics"]
model_name = best_model_results["model_name"]
problem_type = training_results["problem_type"]

with st.sidebar:
    st.markdown("## Export Controls")
    custom_export_name = st.text_input(
        "Custom Export Name",
        value=model_name.replace(" ", "_")
    )

st.markdown("---")
with st.container(border=True):
    st.subheader("Export Summary")
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
    st.subheader("Model Metrics")
    render_metric_row([(metric_name, metric_value) for metric_name, metric_value in metrics.items()])

st.markdown("---")
export_button = st.button("Export Model Pipeline")

if export_button:
    try:
        with st.spinner("Exporting ML pipeline..."):
            export_path = export_model_pipeline(
                pipeline,
                custom_export_name
            )

            metadata = generate_model_metadata(
                model_name=custom_export_name,
                metrics=metrics,
                problem_type=problem_type,
                dataset_shape=dataset.shape,
                feature_names=get_expected_features(pipeline)
            )

            metadata_path = save_model_metadata(
                metadata,
                custom_export_name
            )

            summary = generate_export_summary(
                export_path,
                metadata_path
            )

            st.session_state.export_summary = summary

            st.success("Model exported successfully.")

            st.markdown("---")
            st.subheader("Exported Files")
            st.code(export_path)
            st.code(metadata_path)

            st.markdown("---")
            st.subheader("Download Model")
            with open(export_path, "rb") as model_file:
                st.download_button(
                    label="Download .pkl Model",
                    data=model_file,
                    file_name=os.path.basename(export_path),
                    mime="application/octet-stream"
                )

            with open(metadata_path, "rb") as metadata_file:
                st.download_button(
                    label="Download Metadata JSON",
                    data=metadata_file,
                    file_name=os.path.basename(metadata_path),
                    mime="application/json"
                )

            st.markdown("---")
            st.subheader("Metadata Preview")
            st.json(metadata)

    except Exception as error:
        st.error(f"Export failed: {error}")

if "export_summary" in st.session_state and st.session_state.export_summary is not None:
    st.markdown("---")
    st.success("Export completed successfully.")

render_engineered_data_section(session_state=st.session_state)
