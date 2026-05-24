import pandas as pd
import streamlit as st

from src.evaluation import (
    create_actual_vs_predicted_plot,
    create_precision_recall_chart,
    create_residual_plot,
    create_roc_chart,
    generate_classification_report,
    generate_confusion_matrix,
    generate_precision_recall_curve,
    generate_residual_analysis,
    generate_roc_curve,
)
from src.session_manager import get_missing_or_empty_keys, initialize_session_state
from src.ui_helpers import load_ml_studio_css, render_engineered_data_section, render_metric_row, render_page_header


st.set_page_config(page_title="Model Evaluation | ML Studio", page_icon=":material/monitoring:", layout="wide")
load_ml_studio_css()
initialize_session_state(st.session_state)

render_page_header(
    "Model Evaluation",
    "Evaluate model performance with aligned metrics, diagnostics, and comparison views.",
    "Evaluation",
)

missing_keys = get_missing_or_empty_keys(st.session_state, ["training_results", "best_model", "preprocessing_results"])
if missing_keys:
    st.warning(f"Missing required workflow artifacts: {', '.join(missing_keys)}")
    st.stop()

training_results = st.session_state.training_results
best_model_results = st.session_state.best_model
preprocessing_results = st.session_state.preprocessing_results
problem_type = training_results["problem_type"]
try:
    pipeline = best_model_results["pipeline"]
    X_test = preprocessing_results.get("X_test_raw", preprocessing_results["X_test"])
    y_test = preprocessing_results["y_test"]
    predictions = pipeline.predict(X_test)
except Exception as error:
    st.error(f"Unable to evaluate the active model: {error}")
    st.stop()

available_analyses = ["Performance Metrics"]
if problem_type == "classification":
    available_analyses.extend(["Confusion Matrix", "ROC Curve", "Precision Recall Curve"])
else:
    available_analyses.extend(["Residual Analysis", "Actual vs Predicted"])

with st.sidebar:
    st.markdown("## Evaluation Controls")
    selected_analysis = st.radio("Select Analysis", options=available_analyses)

if selected_analysis == "Performance Metrics":
    st.markdown("---")
    with st.container(border=True):
        st.subheader("Model Metrics")
        metrics = best_model_results["metrics"]
        render_metric_row([(metric_name, metric_value) for metric_name, metric_value in metrics.items()])

elif selected_analysis == "Confusion Matrix":
    st.markdown("---")
    st.subheader("Confusion Matrix")
    matrix, labels = generate_confusion_matrix(y_test, predictions)
    st.dataframe(pd.DataFrame(matrix, index=labels, columns=labels), width="stretch")
    st.markdown("---")
    st.subheader("Classification Report")
    st.dataframe(generate_classification_report(y_test, predictions), width="stretch")

elif selected_analysis in {"ROC Curve", "Precision Recall Curve"}:
    st.markdown("---")
    st.subheader("ROC Curve" if selected_analysis == "ROC Curve" else "Precision Recall Curve")
    if not hasattr(pipeline, "predict_proba"):
        st.warning("This model does not expose class probabilities.")
    else:
        try:
            probabilities = pipeline.predict_proba(X_test)[:, 1]
            if selected_analysis == "ROC Curve":
                st.plotly_chart(create_roc_chart(generate_roc_curve(y_test, probabilities)), width="stretch")
            else:
                st.plotly_chart(
                    create_precision_recall_chart(generate_precision_recall_curve(y_test, probabilities)),
                    width="stretch",
                )
        except Exception as error:
            st.warning(str(error))

elif selected_analysis == "Residual Analysis":
    st.markdown("---")
    st.subheader("Residual Analysis")
    residual_dataframe = generate_residual_analysis(y_test, predictions)
    st.dataframe(residual_dataframe, width="stretch")
    st.plotly_chart(create_residual_plot(residual_dataframe), width="stretch")

elif selected_analysis == "Actual vs Predicted":
    st.markdown("---")
    st.subheader("Actual vs Predicted")
    st.plotly_chart(create_actual_vs_predicted_plot(y_test, predictions), width="stretch")

render_engineered_data_section(session_state=st.session_state)
st.markdown("---")
st.success("Model evaluation completed")
