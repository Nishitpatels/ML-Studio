# =========================================================
# FILE: pages/18_Chat_With_Dataset.py
# =========================================================


from html import escape

import streamlit as st

from src.ai_assistant import (
    chat_with_dataset,
    get_ai_configuration_error,
    get_ai_model_label,
)
from src.session_manager import get_modeling_dataset, initialize_session_state
from src.ui_helpers import load_ml_studio_css, render_engineered_data_section, render_page_header


st.set_page_config(
    page_title="Chat With Dataset | ML Studio",
    page_icon=":material/chat:",
    layout="wide",
)

load_ml_studio_css()
render_page_header(
    "Chat With Dataset",
    "Ask AI questions about the dataset currently active in the workflow.",
    "Dataset Chat",
)

initialize_session_state(st.session_state)
dataset = get_modeling_dataset(st.session_state)
if dataset is None:
    st.warning("Please upload a dataset first.")
    st.stop()

ai_error = get_ai_configuration_error()
ai_enabled = ai_error is None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


def render_chat_bubble(role: str, content: str) -> None:
    row_class = "ml-chat-row-user" if role == "user" else "ml-chat-row-assistant"
    bubble_class = "ml-chat-user" if role == "user" else "ml-chat-assistant"
    safe_content = escape(content).replace("\n", "<br>")
    st.markdown(
        f"""
        <div class="{row_class}">
            <div class="{bubble_class}">{safe_content}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with st.sidebar:
    st.markdown("## Suggested Questions")
    st.caption(f"Model: {get_ai_model_label()}")
    suggested_questions = [
        "Which columns may cause overfitting?",
        "What preprocessing is recommended?",
        "Which features are most important?",
        "Does this dataset have quality issues?",
        "What ML models are best for this data?",
        "What feature engineering can improve performance?",
    ]
    for question in suggested_questions:
        st.info(question)

if not ai_enabled:
    st.warning(f"Gemini is unavailable: {ai_error}")
    st.info("Add a valid API key or supported model in `.env`, then refresh the app.")

st.markdown("---")
with st.container(border=True):
    st.subheader("Conversation")
    if not st.session_state.chat_history:
        st.info("Start a dataset conversation using the input below.")
    for message in st.session_state.chat_history:
        if not isinstance(message, dict) or "role" not in message or "content" not in message:
            continue
        render_chat_bubble(message["role"], message["content"])

user_question = st.chat_input(
    "Ask anything about your dataset...",
    disabled=not ai_enabled,
)

if user_question:
    st.session_state.chat_history.append(
        {
            "role": "user",
            "content": user_question,
        }
    )

    render_chat_bubble("user", user_question)

    with st.spinner("AI analyzing dataset..."):
        try:
            response = chat_with_dataset(
                dataframe=dataset,
                user_question=user_question,
                chat_history=st.session_state.chat_history,
            )
            render_chat_bubble("assistant", response)
            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": response,
                }
            )
        except Exception as error:
            st.error(f"AI chat failed: {error}")

st.markdown("---")
if st.button("Clear Chat History"):
    st.session_state.chat_history = []
    st.rerun()

render_engineered_data_section(session_state=st.session_state)
