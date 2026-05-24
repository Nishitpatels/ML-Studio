# =========================================================
# FILE: pages/13_AI_Assistant.py
# =========================================================


import streamlit as st

from src.ai_assistant import (
    ask_ai_question,
    generate_dataset_summary,
    generate_eda_insights,
    generate_model_recommendations,
    get_ai_configuration_error,
    get_ai_model_label,
)
from src.session_manager import get_modeling_dataset, initialize_session_state
from src.ui_helpers import load_ml_studio_css, render_engineered_data_section, render_page_header


st.set_page_config(
    page_title="AI Assistant | ML Studio",
    page_icon=":material/smart_toy:",
    layout="wide",
)

load_ml_studio_css()
render_page_header(
    "AI ML Assistant",
    "AI-powered dataset and ML guidance for the dataset active in the workflow.",
    "Assistant",
)
initialize_session_state(st.session_state)

dataset = get_modeling_dataset(st.session_state)
if dataset is None:
    st.warning("Please upload a dataset first.")
    st.stop()

target_column = st.session_state.get("target_column")
ai_error = get_ai_configuration_error()
ai_enabled = ai_error is None

with st.sidebar:
    st.markdown("## AI Assistant Options")
    st.caption(f"Model: {get_ai_model_label()}")
    selected_option = st.radio(
        "Select AI Feature",
        options=[
            "Dataset Summary",
            "EDA Insights",
            "Model Recommendations",
            "Ask Anything",
        ],
    )

if not ai_enabled:
    st.warning(f"Gemini is unavailable: {ai_error}")
    st.info("Add a valid API key or supported model in `.env`, then refresh the app.")

if selected_option == "Dataset Summary":
    st.markdown("---")
    st.subheader("AI Dataset Summary")
    if st.button("Generate AI Summary", disabled=not ai_enabled):
        with st.spinner("AI analyzing dataset..."):
            try:
                summary = generate_dataset_summary(dataset)
                st.success("AI analysis completed.")
                st.markdown(summary)
            except Exception as error:
                st.error(f"AI analysis failed: {error}")

elif selected_option == "EDA Insights":
    st.markdown("---")
    st.subheader("AI EDA Insights")
    if st.button("Generate EDA Insights", disabled=not ai_enabled):
        with st.spinner("AI generating EDA insights..."):
            try:
                insights = generate_eda_insights(dataset)
                st.success("AI insights generated.")
                st.markdown(insights)
            except Exception as error:
                st.error(f"EDA insights failed: {error}")

elif selected_option == "Model Recommendations":
    st.markdown("---")
    st.subheader("AI Model Recommendations")
    if target_column is None or target_column not in dataset.columns:
        st.warning("Please select a target column that exists in the current dataset.")
    elif st.button("Generate Recommendations", disabled=not ai_enabled):
        with st.spinner("AI recommending models..."):
            try:
                recommendations = generate_model_recommendations(
                    dataset,
                    target_column,
                )
                st.success("Recommendations generated.")
                st.markdown(recommendations)
            except Exception as error:
                st.error(f"Recommendation failed: {error}")

elif selected_option == "Ask Anything":
    st.markdown("---")
    st.subheader("Ask AI Anything")
    user_question = st.text_area(
        "Ask your ML/Data Science question",
        placeholder="Example: Which features may cause overfitting?",
        disabled=not ai_enabled,
    )
    if st.button("Ask AI", disabled=not ai_enabled):
        if user_question.strip() == "":
            st.warning("Please enter a question.")
        else:
            with st.spinner("AI thinking..."):
                try:
                    answer = ask_ai_question(user_question, dataset)
                    st.success("AI response generated.")
                    st.markdown(answer)
                except Exception as error:
                    st.error(f"AI question failed: {error}")

render_engineered_data_section(session_state=st.session_state)
st.markdown("---")
st.success("AI Assistant ready.")
