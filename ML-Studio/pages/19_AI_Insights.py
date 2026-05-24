# =========================================================
# FILE: pages/19_AI_Insights.py
# =========================================================


import streamlit as st

from src.ai_assistant import (
    generate_ai_insights,
    get_ai_configuration_error,
    get_ai_model_label,
)
from src.session_manager import get_modeling_dataset, initialize_session_state
from src.ui_helpers import load_ml_studio_css, render_engineered_data_section, render_metric_row, render_page_header


st.set_page_config(
    page_title="AI Insights | ML Studio",
    page_icon=":material/psychology:",
    layout="wide",
)

load_ml_studio_css()
render_page_header(
    "AI Insights Engine",
    "Generate advanced AI insights for the dataset currently active in the workflow.",
    "AI Insights",
)

initialize_session_state(st.session_state)
dataset = get_modeling_dataset(st.session_state)
if dataset is None:
    st.warning("Please upload a dataset first.")
    st.stop()

ai_error = get_ai_configuration_error()
ai_enabled = ai_error is None

with st.sidebar:
    st.markdown("## Insight Controls")
    st.caption(f"Model: {get_ai_model_label()}")
    st.success("AI will generate advanced ML and business insights.")

if not ai_enabled:
    st.warning(f"Gemini is unavailable: {ai_error}")
    st.info("Add a valid API key or supported model in `.env`, then refresh the app.")

st.markdown("---")
with st.container(border=True):
    st.subheader("Dataset Overview")
    render_metric_row(
        [
            ("Rows", dataset.shape[0]),
            ("Columns", dataset.shape[1]),
            ("Missing Values", int(dataset.isna().sum().sum())),
            ("Duplicate Rows", int(dataset.duplicated().sum())),
        ],
        columns=4,
    )

st.markdown("---")
generate_button = st.button(
    "Generate AI Insights",
    disabled=not ai_enabled,
)

if generate_button:
    with st.spinner("AI generating advanced insights..."):
        try:
            insights = generate_ai_insights(dataset)
            st.success("AI insights generated.")
            st.markdown("---")
            st.subheader("AI Analysis Report")
            st.markdown(insights)
        except Exception as error:
            st.error(f"AI insights failed: {error}")

render_engineered_data_section(session_state=st.session_state)
st.markdown("---")
st.success("AI Insights system ready.")
