# =========================================================
# FILE: pages/10_Explainability.py
# =========================================================


import pandas as pd
import streamlit as st

from src.explainability import (
    create_feature_importance_chart,
    create_shap_importance_chart,
    explain_single_prediction,
    generate_explainability_summary,
    generate_shap_summary_dataframe,
    generate_shap_values,
    get_feature_importance,
    get_transformed_feature_names,
)
from src.session_manager import get_missing_or_empty_keys, initialize_session_state
from src.ui_helpers import load_ml_studio_css, render_engineered_data_section, render_metric_row, render_page_header
from src.visualization import plot_metric_bar_chart


st.set_page_config(
    page_title="Explainability | ML Studio",
    page_icon=":material/psychology:",
    layout="wide",
)

load_ml_studio_css()
render_page_header(
    "Explainability Dashboard",
    "Understand how the best fitted model is behaving on the active workflow dataset.",
    "Model Interpretation",
)
initialize_session_state(st.session_state)

required_keys = ["training_results", "best_model", "preprocessing_results", "dataset", "target_column"]
missing_keys = get_missing_or_empty_keys(st.session_state, required_keys)
if missing_keys:
    st.warning(f"Missing required workflow artifacts: {', '.join(missing_keys)}")
    st.stop()

training_results = st.session_state.training_results
best_model_results = st.session_state.best_model
preprocessing_results = st.session_state.preprocessing_results

pipeline = best_model_results["pipeline"]
X_train = preprocessing_results.get("X_train_raw", preprocessing_results["X_train"])
X_test = preprocessing_results.get("X_test_raw", preprocessing_results["X_test"])
problem_type = training_results["problem_type"]
best_model_name = training_results.get("best_model_name") or best_model_results.get("model_name")
best_metrics = best_model_results["metrics"]

with st.sidebar:
    st.markdown("## Explainability Controls")
    selected_analysis = st.radio(
        "Select Analysis",
        options=[
            "Model Summary",
            "Feature Importance",
            "SHAP Explainability",
            "Prediction Explanation",
        ],
    )

with st.container(border=True):
    st.success(f"Best Fitted Model: {best_model_name}")
    render_metric_row(
        [
            ("Problem Type", problem_type.title()),
            ("Training Rows", X_train.shape[0]),
            ("Processed Features", X_train.shape[1]),
        ],
        columns=3,
    )

if selected_analysis == "Model Summary":
    st.markdown("---")
    st.subheader("Explainability Summary")

    summary = generate_explainability_summary(pipeline, X_train)

    with st.container(border=True):
        render_metric_row(
            [
                ("Model", summary["Model Name"]),
                ("Feature Importance", str(summary["Supports Feature Importance"])),
                ("SHAP Support", str(summary["Supports SHAP"])),
                ("Total Features", summary["Total Features"]),
            ],
            columns=4,
        )

    st.markdown("---")
    with st.container(border=True):
        st.subheader("Best Model Metrics")
        render_metric_row([(metric_name, metric_value) for metric_name, metric_value in best_metrics.items()])

    metrics_chart = plot_metric_bar_chart(
        best_metrics,
        "Best Model Metric Overview",
    )
    if metrics_chart is not None:
        st.plotly_chart(metrics_chart, width="stretch")

elif selected_analysis == "Feature Importance":
    st.markdown("---")
    st.subheader("Feature Importance Analysis")
    try:
        feature_importance_dataframe = get_feature_importance(pipeline, X_train)
        if feature_importance_dataframe is None:
            st.warning("Feature importance is not supported for this model.")
        else:
            feature_importance_chart = create_feature_importance_chart(
                feature_importance_dataframe
            )
            table_col, chart_col = st.columns([1.0, 1.45], gap="large")
            with table_col:
                with st.container(border=True):
                    st.caption("Ranked importance table")
                    st.dataframe(feature_importance_dataframe, width="stretch", hide_index=True)
            with chart_col:
                st.plotly_chart(feature_importance_chart, width="stretch")
    except Exception as error:
        st.error(f"Feature importance failed: {error}")

elif selected_analysis == "SHAP Explainability":
    st.markdown("---")
    st.subheader("SHAP Explainability")
    try:
        with st.spinner("Generating SHAP values..."):
            shap_values = generate_shap_values(pipeline, X_train)
            if shap_values is None:
                st.warning("SHAP explainability is not supported for this model.")
            else:
                feature_names = get_transformed_feature_names(pipeline, X_train)
                shap_dataframe = generate_shap_summary_dataframe(
                    shap_values,
                    feature_names,
                )
                shap_chart = create_shap_importance_chart(shap_dataframe)
                table_col, chart_col = st.columns([1.0, 1.45], gap="large")
                with table_col:
                    with st.container(border=True):
                        st.caption("Mean absolute SHAP impact")
                        st.dataframe(shap_dataframe, width="stretch", hide_index=True)
                with chart_col:
                    st.plotly_chart(shap_chart, width="stretch")
    except Exception as error:
        st.error(f"SHAP explainability failed: {error}")

elif selected_analysis == "Prediction Explanation":
    st.markdown("---")
    st.subheader("Prediction Explanation")
    prediction_index = st.slider(
        "Select Prediction Index",
        min_value=0,
        max_value=max(len(X_test) - 1, 0),
        value=0,
    )
    try:
        with st.spinner("Generating prediction explanation..."):
            prediction_explanation = explain_single_prediction(
                pipeline,
                X_test,
                row_index=prediction_index,
            )
            if prediction_explanation is None:
                st.warning("Prediction explanation is unavailable for this model.")
            else:
                feature_names = get_transformed_feature_names(pipeline, X_train)
                explanation_dataframe = pd.DataFrame(
                    {
                        "Feature": feature_names,
                        "Contribution": prediction_explanation.values,
                    }
                ).sort_values(by="Contribution", ascending=False)

                st.dataframe(explanation_dataframe, width="stretch", hide_index=True)

                contribution_chart = explanation_dataframe.head(20).set_index("Feature")
                st.bar_chart(contribution_chart)

                st.markdown("---")
                st.subheader("Selected Model Input Row")
                st.dataframe(X_test.iloc[[prediction_index]], width="stretch", hide_index=True)
    except Exception as error:
        st.error(f"Prediction explanation failed: {error}")

render_engineered_data_section(session_state=st.session_state)
st.markdown("---")
st.success("Explainability analysis completed.")
