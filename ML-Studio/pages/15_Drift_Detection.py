# =========================================================
# FILE: pages/15_Drift_Detection.py
# =========================================================


import pandas as pd
import streamlit as st

from src.drift_detection import (
    create_drift_chart,
    detect_feature_drift,
    generate_drift_summary,
)
from src.session_manager import get_modeling_dataset, initialize_session_state
from src.ui_helpers import load_ml_studio_css, render_engineered_data_section, render_metric_row, render_page_header


st.set_page_config(
    page_title="Drift Detection | ML Studio",
    page_icon=":material/trending_down:",
    layout="wide"
)

load_ml_studio_css()
render_page_header(
    "Data Drift Detection",
    "Compare current data against the active training dataset and review drift severity.",
    "Monitoring",
)
initialize_session_state(st.session_state)

reference_dataset = get_modeling_dataset(st.session_state)
if reference_dataset is None:
    st.warning("Please upload training dataset first.")
    st.stop()

with st.sidebar:
    st.markdown("## Drift Detection Settings")
    st.info("Upload a new dataset to compare against training data.")

st.markdown("---")
st.subheader("Upload Current Dataset")

uploaded_current_dataset = st.file_uploader(
    "Upload Current Dataset CSV",
    type=["csv"]
)

if uploaded_current_dataset is not None:
    try:
        current_dataset = pd.read_csv(uploaded_current_dataset)

        st.markdown("---")
        st.subheader("Current Dataset Preview")
        st.dataframe(current_dataset.head(), width="stretch")

        with st.spinner("Detecting data drift..."):
            drift_dataframe = detect_feature_drift(
                reference_dataset,
                current_dataset
            )

            st.session_state.drift_results = drift_dataframe

            st.success("Drift analysis completed.")
            if drift_dataframe.empty:
                st.info("No comparable numeric features were available for drift analysis.")

            st.markdown("---")
            with st.container(border=True):
                st.subheader("Drift Summary")
                summary = generate_drift_summary(drift_dataframe)
                render_metric_row(
                    [
                        ("Total Features", summary["Total Features"]),
                        ("No Drift", summary["No Drift Features"]),
                        ("Moderate Drift", summary["Moderate Drift Features"]),
                        ("Significant Drift", summary["Significant Drift Features"]),
                    ],
                    columns=4,
                )

            st.markdown("---")
            st.subheader("Feature Drift Results")
            st.dataframe(drift_dataframe, width="stretch")

            st.markdown("---")
            st.subheader("Drift Visualization")
            drift_chart = create_drift_chart(drift_dataframe)
            st.plotly_chart(drift_chart, width="stretch")

            st.markdown("---")
            st.subheader("Significant Drift Features")
            significant_drift = drift_dataframe[
                drift_dataframe["Drift Status"] == "Significant Drift"
            ]

            if len(significant_drift) == 0:
                st.success("No significant drift detected.")
            else:
                st.warning("Significant drift detected in some features.")
                st.dataframe(significant_drift, width="stretch")

    except Exception as error:
        st.error(f"Drift detection failed: {error}")

render_engineered_data_section(session_state=st.session_state)
st.markdown("---")
st.success("Drift detection system ready.")
