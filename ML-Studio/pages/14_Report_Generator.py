# =========================================================
# FILE: pages/14_Report_Generator.py
# =========================================================


import streamlit as st

from src.explainability import generate_explainability_summary
from src.report_generator import generate_complete_report
from src.session_manager import get_missing_or_empty_keys, get_modeling_dataset, initialize_session_state
from src.ui_helpers import load_ml_studio_css, render_engineered_data_section, render_page_header


st.set_page_config(
    page_title="Report Generator | ML Studio",
    page_icon=":material/description:",
    layout="wide"
)

load_ml_studio_css()
render_page_header(
    "Automated Report Generator",
    "Generate a professional PDF report from the current ML Studio workflow state.",
    "Reporting",
)
initialize_session_state(st.session_state)

required_keys = [
    "dataset",
    "training_results",
    "best_model"
]

missing_keys = get_missing_or_empty_keys(st.session_state, required_keys)
if missing_keys:
    st.warning(f"Missing required workflow artifacts: {', '.join(missing_keys)}")
    st.stop()

dataset = get_modeling_dataset(st.session_state)
training_results = st.session_state.training_results
best_model_results = st.session_state.best_model
preprocessing_results = st.session_state.get("preprocessing_results", None)

with st.sidebar:
    st.markdown("## Report Settings")
    include_explainability = st.checkbox(
        "Include Explainability",
        value=preprocessing_results is not None,
        disabled=preprocessing_results is None
    )
    include_preprocessing = st.checkbox(
        "Include Preprocessing",
        value=preprocessing_results is not None,
        disabled=preprocessing_results is None
    )

pipeline = best_model_results["pipeline"]

explainability_summary = None
if include_explainability and preprocessing_results is not None:
    try:
        explainability_summary = generate_explainability_summary(
            pipeline,
            preprocessing_results["X_train"]
        )
    except Exception as error:
        st.warning(f"Explainability summary is unavailable: {error}")

st.markdown("---")
st.subheader("Report Contents")

report_sections = [
    "Dataset Summary",
    "Training Summary",
    "Model Metrics"
]

if include_preprocessing:
    report_sections.append("Preprocessing Summary")

if include_explainability:
    report_sections.append("Explainability Summary")

for section in report_sections:
    st.success(section)

st.info("The PDF uses a cleaner report layout with improved spacing, borders, table formatting, and Times-style typography.")

st.markdown("---")
generate_report_button = st.button("Generate PDF Report")

if generate_report_button:
    try:
        with st.spinner("Generating PDF report..."):
            report_path = generate_complete_report(
                dataset=dataset,
                training_results=training_results,
                best_model_results=best_model_results,
                preprocessing_results=preprocessing_results if include_preprocessing else None,
                explainability_summary=(
                    explainability_summary
                    if include_explainability and explainability_summary is not None
                    else None
                ),
            )

            st.success("Report generated successfully.")

            st.markdown("---")
            st.subheader("Report Path")
            st.code(report_path)

            st.markdown("---")
            st.subheader("Download Report")
            with open(report_path, "rb") as report_file:
                st.download_button(
                    label="Download PDF Report",
                    data=report_file,
                    file_name="ML_Studio_Report.pdf",
                    mime="application/pdf"
                )

    except Exception as error:
        st.error(f"Report generation failed: {error}")

render_engineered_data_section(session_state=st.session_state)
st.markdown("---")
st.success("Report generator ready.")
