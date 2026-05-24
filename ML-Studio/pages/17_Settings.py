import platform

import pandas as pd
import streamlit as st

from src.settings_manager import (
    clear_directory,
    clear_session_state,
    generate_storage_summary,
    initialize_directories,
)
from src.ui_helpers import load_ml_studio_css, render_metric_row, render_page_header


st.set_page_config(
    page_title="Settings | ML Studio",
    page_icon=":material/settings:",
    layout="wide",
)

load_ml_studio_css()
initialize_directories()

render_page_header(
    "ML Studio Settings",
    "Manage session state, workspace storage, and local system information.",
    "Settings",
)

with st.sidebar:
    st.markdown("### Settings Navigation")
    selected_section = st.radio(
        "Select Section",
        options=[
            "System Overview",
            "Session Management",
            "Storage Management",
            "System Information",
        ],
    )

if selected_section == "System Overview":
    st.markdown("---")
    with st.container(border=True):
        st.subheader("System Storage Summary")
        storage_summary = generate_storage_summary()
        render_metric_row([(name, count) for name, count in storage_summary.items()])

    st.markdown("---")
    with st.container(border=True):
        st.subheader("Active Session Variables")
        session_keys = list(st.session_state.keys())
        if not session_keys:
            st.info("No active session variables.")
        else:
            st.dataframe(pd.DataFrame({"Session Key": session_keys}), width="stretch", hide_index=True)

elif selected_section == "Session Management":
    st.markdown("---")
    with st.container(border=True):
        st.subheader("Session Management")
        st.warning("Clearing session will reset uploaded datasets, workflow artifacts, and model state.")
        if st.button("Clear Session State", width="stretch"):
            clear_session_state(st.session_state)
            st.success("Session state cleared.")

elif selected_section == "Storage Management":
    storage_options = {
        "Reports": "reports",
        "Experiments": "experiments",
        "Exported Models": "models/exported",
        "Metadata": "models/metadata",
    }

    st.markdown("---")
    with st.container(border=True):
        st.subheader("Storage Management")
        summary = generate_storage_summary()
        st.dataframe(
            pd.DataFrame(
                {
                    "Storage Area": list(summary.keys()),
                    "Files": list(summary.values()),
                }
            ),
            width="stretch",
            hide_index=True,
        )

        selected_storage = st.selectbox(
            "Select Storage Area to Clear",
            options=list(storage_options.keys()),
        )
        if st.button("Clear Selected Storage", width="stretch"):
            directory_path = storage_options[selected_storage]
            success = clear_directory(directory_path)
            if success:
                st.success(f"Cleared {selected_storage}.")
            else:
                st.error("Storage clearing failed.")

elif selected_section == "System Information":
    system_info = {
        "Platform": platform.system(),
        "Platform Version": platform.version(),
        "Processor": platform.processor(),
        "Python Version": platform.python_version(),
    }

    st.markdown("---")
    with st.container(border=True):
        st.subheader("System Information")
        st.dataframe(
            pd.DataFrame(
                {
                    "Property": list(system_info.keys()),
                    "Value": list(system_info.values()),
                }
            ),
            width="stretch",
            hide_index=True,
        )

    st.markdown("---")
    with st.container(border=True):
        st.subheader("ML Studio Information")
        st.success("ML Studio v1.0")
        st.info(
            "Included modules: AutoML, preprocessing, explainability, drift detection, "
            "experiment tracking, AI assistant, PDF reports, and model export."
        )

st.markdown("---")
st.success("Settings system ready.")
