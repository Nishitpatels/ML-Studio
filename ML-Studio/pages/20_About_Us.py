import pandas as pd
import streamlit as st

from src.ui_helpers import (
    load_ml_studio_css,
    render_dashboard_card,
    render_metric_row,
    render_page_header,
)


st.set_page_config(page_title="About Us | ML Studio", page_icon=":material/info:", layout="wide")
load_ml_studio_css()

render_page_header(
    "About Us",
    "A concise overview of ML Studio, its stack, creator, support details, and roadmap.",
    "Platform Information",
)

st.markdown("---")
overview_col, version_col = st.columns([1.55, 1.0], gap="large")

with overview_col:
    with st.container(border=True):
        st.subheader("Platform Overview")
        st.write(
            "ML Studio is an AI-assisted machine learning platform designed for preprocessing, "
            "feature engineering, training, evaluation, explainability, prediction, and workflow automation."
        )

with version_col:
    with st.container(border=True):
        st.subheader("Version Information")
        render_metric_row(
            [
                ("Product", "ML Studio"),
                ("Version", "v1.0"),
            ],
            columns=2,
        )

st.markdown("---")
stack_col, creator_col = st.columns(2, gap="large")

with stack_col:
    with st.container(border=True):
        st.subheader("Tech Stack")
        stack = pd.DataFrame(
            {
                "Layer": [
                    "Application UI",
                    "Data Workflow",
                    "Machine Learning",
                    "Visualization",
                    "AI Assistance",
                    "Reports and Export",
                ],
                "Tools": [
                    "Streamlit",
                    "Pandas",
                    "Scikit-learn",
                    "Plotly, Seaborn, Matplotlib",
                    "Gemini integration",
                    "ReportLab, Joblib, JSON metadata",
                ],
            }
        )
        st.dataframe(stack, width="stretch", hide_index=True)

with creator_col:
    with st.container(border=True):
        st.subheader("Creator Section")
        st.markdown("**Developed by Nishit Patel**")
        st.markdown("GitHub: [Nishitpatels](https://github.com/Nishitpatels)")
        st.markdown("Email: [support.mlstudio@gmail.com](mailto:support.mlstudio@gmail.com)")
        st.markdown("LinkedIn: [Nishit Patel](https://www.linkedin.com/in/nishit-patel-2b2045296/)")

st.markdown("---")
support_col, roadmap_col = st.columns(2, gap="large")

with support_col:
    with st.container(border=True):
        st.subheader("Contact and Support")
        st.write("For feedback or collaboration:")
        st.markdown("[support.mlstudio@gmail.com](mailto:support.mlstudio@gmail.com)")

with roadmap_col:
    with st.container(border=True):
        st.subheader("Future Roadmap")
        roadmap_items = [
            "Deployment APIs",
            "Docker Support",
            "MLflow Integration",
            "Advanced AutoML",
        ]
        for item in roadmap_items:
            st.info(item)

st.markdown("---")
feature_col1, feature_col2, feature_col3 = st.columns(3)
with feature_col1:
    render_dashboard_card("Workflow First", "Built around practical ML workflow continuity from dataset upload to export.")
with feature_col2:
    render_dashboard_card("Transparent Controls", "Keeps preprocessing, training, and tuning choices visible and reviewable.")
with feature_col3:
    render_dashboard_card("Analyst Friendly", "Designed for analysts, students, and ML practitioners who need a clear workspace.")
