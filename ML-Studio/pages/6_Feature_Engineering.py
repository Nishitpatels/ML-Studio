import pandas as pd
import streamlit as st

from src.feature_engineering import (
    apply_ai_recommended_features,
    apply_feature_binning,
    apply_log_transformation,
    create_interaction_feature,
    create_polynomial_features,
    detect_datetime_columns,
    extract_datetime_features,
    generate_feature_engineering_suggestions,
    generate_feature_engineering_summary,
    get_ai_feature_recommendations,
    get_categorical_columns,
    get_numerical_columns,
)
from src.session_manager import (
    apply_workflow_stage_result,
    get_workflow_stage_dataset,
    get_workflow_stage_payload,
    initialize_session_state,
    reset_workflow_stage,
    set_preprocessing_results,
    set_working_dataset,
    sync_workflow_stage,
    undo_workflow_stage,
)
from src.ui_helpers import load_ml_studio_css, render_engineered_data_section, render_metric_row, render_page_header


st.set_page_config(
    page_title="Feature Engineering | ML Studio",
    page_icon=":material/psychology:",
    layout="wide",
)

load_ml_studio_css()
render_page_header(
    "Feature Engineering Lab",
    "Add new features manually, keep changes reversible, and rerun preprocessing only when needed.",
    "Feature Engineering",
)
initialize_session_state(st.session_state)

if st.session_state.dataset is None:
    st.warning("No dataset found. Please upload a dataset first.")
    st.stop()


def restore_feature_engineering_state(restored_dataset: pd.DataFrame, restored_payload: dict | None):
    if restored_payload is not None:
        set_working_dataset(st.session_state, restored_dataset, engineered=True)
        st.session_state.engineered_dataset = restored_dataset.copy()
        return

    preprocessing_payload = get_workflow_stage_payload(st.session_state, "preprocessing") or {}
    if preprocessing_payload.get("preprocessing_results") is not None:
        set_working_dataset(st.session_state, restored_dataset, engineered=False)
        set_preprocessing_results(
            st.session_state,
            preprocessing_payload["preprocessing_results"],
            preprocessing_payload.get("preprocessing_config"),
        )
        st.session_state.engineered_dataset = None
        return

    set_working_dataset(st.session_state, restored_dataset, engineered=False)
    st.session_state.engineered_dataset = None


def apply_feature_engineering_result(updated_dataframe: pd.DataFrame, operation_name: str):
    previous_columns = list(current_feature_dataset.columns)
    new_columns = [column for column in updated_dataframe.columns if column not in previous_columns]

    set_working_dataset(st.session_state, updated_dataframe, engineered=True)
    st.session_state.engineered_dataset = updated_dataframe.copy()
    apply_workflow_stage_result(
        st.session_state,
        "feature_engineering",
        updated_dataframe,
        payload={
            "operation": operation_name,
            "new_columns": new_columns,
        },
    )

    if new_columns:
        st.success(f"New columns added: {', '.join(new_columns)}")
    else:
        st.success("Feature engineering changes applied successfully.")

    st.info("Preprocessing has been cleared for safety. Rerun preprocessing before training.")
    st.dataframe(updated_dataframe.head(), width="stretch")


preprocessing_stage_dataset = get_workflow_stage_dataset(st.session_state, "preprocessing")
feature_source_dataset = (
    st.session_state.engineered_dataset.copy()
    if st.session_state.engineered_dataset is not None
    else preprocessing_stage_dataset.copy()
    if preprocessing_stage_dataset is not None
    else st.session_state.working_dataset.copy()
    if st.session_state.working_dataset is not None
    else st.session_state.dataset.copy()
)

stage_state = sync_workflow_stage(st.session_state, "feature_engineering", feature_source_dataset)
source_feature_dataset = get_workflow_stage_dataset(
    st.session_state,
    "feature_engineering",
    snapshot_name="previous_stage_dataset",
)
current_feature_dataset = get_workflow_stage_dataset(st.session_state, "feature_engineering")
target_column = st.session_state.target_column

numerical_columns = get_numerical_columns(current_feature_dataset)
if target_column in numerical_columns:
    numerical_columns.remove(target_column)

categorical_columns = get_categorical_columns(current_feature_dataset)
if target_column in categorical_columns:
    categorical_columns.remove(target_column)

datetime_columns = detect_datetime_columns(current_feature_dataset)
if target_column in datetime_columns:
    datetime_columns.remove(target_column)

top_col1, top_col2 = st.columns([3, 2])
with top_col1:
    with st.container(border=True):
        st.subheader("Workflow Continuity")
        render_metric_row(
            [
                (
                    "Original Snapshot",
                    f"{stage_state['original_dataset'].shape[0]} x {stage_state['original_dataset'].shape[1]}",
                ),
                ("Previous Stage Snapshot", f"{source_feature_dataset.shape[0]} x {source_feature_dataset.shape[1]}"),
                ("Current Page Snapshot", f"{current_feature_dataset.shape[0]} x {current_feature_dataset.shape[1]}"),
            ],
            columns=3,
        )
with top_col2:
    with st.container(border=True):
        st.subheader("Page Controls")
        undo_col, reset_col = st.columns(2)
        if undo_col.button("Undo Feature Engineering", width="stretch"):
            restored_dataset, restored_payload = undo_workflow_stage(st.session_state, "feature_engineering")
            restore_feature_engineering_state(restored_dataset, restored_payload)
            st.success("Previous feature engineering state restored.")
            st.rerun()
        if reset_col.button("Reset Current Page Changes", width="stretch"):
            restored_dataset, restored_payload = reset_workflow_stage(st.session_state, "feature_engineering")
            restore_feature_engineering_state(restored_dataset, restored_payload)
            st.success("Feature engineering page reset to its incoming dataset.")
            st.rerun()

with st.sidebar:
    st.markdown("## Feature Engineering Options")
    selected_operation = st.radio(
        "Select Operation",
        options=[
            "Dataset Summary",
            "AI Recommended Features",
            "Interaction Features",
            "Polynomial Features",
            "Log Transformation",
            "Feature Binning",
            "Datetime Features",
        ],
    )

st.markdown("---")
with st.container(border=True):
    render_metric_row(
        [
            ("Rows", current_feature_dataset.shape[0]),
            ("Columns", current_feature_dataset.shape[1]),
            ("Target Column", target_column or "Not Selected"),
            (
                "Workflow Stage",
                "Needs Preprocessing" if not st.session_state.preprocessing_completed else "Preprocessing Ready",
            ),
        ],
        columns=4,
    )

if selected_operation == "Dataset Summary":
    with st.container(border=True):
        st.subheader("Feature Engineering Summary")
        summary = generate_feature_engineering_summary(current_feature_dataset)
        render_metric_row(
            [
                ("Rows", summary["total_rows"]),
                ("Columns", summary["total_columns"]),
                ("Skewed Features", summary["skewed_features"]),
                ("Numerical Columns", summary["numerical_columns"]),
                ("Categorical Columns", summary["categorical_columns"]),
                ("Datetime Columns", summary["datetime_columns"]),
            ],
            columns=3,
        )

    st.markdown("---")
    with st.container(border=True):
        st.subheader("ML Studio Suggestions")
        suggestions = generate_feature_engineering_suggestions(current_feature_dataset)
        if suggestions:
            for suggestion in suggestions:
                st.info(suggestion)
        else:
            st.success("No additional feature engineering suggestions were detected.")

elif selected_operation == "AI Recommended Features":
    st.subheader("AI Recommended Features")
    st.caption("Review the recommendations, then choose only the ones you want to apply.")
    recommendations = get_ai_feature_recommendations(current_feature_dataset)

    if not recommendations:
        st.info("No guided feature recommendations are available for the current dataset.")
    else:
        selected_recommendations: list[str] = []
        for recommendation in recommendations:
            selected = st.checkbox(
                recommendation["label"],
                key=f"feature_recommendation_{recommendation['id']}",
            )
            description_col, benefit_col = st.columns(2)
            description_col.caption(f"Why: {recommendation['description']}")
            benefit_col.caption(f"Expected benefit: {recommendation['expected_benefit']}")
            if selected:
                selected_recommendations.append(recommendation["id"])

        if st.button("Apply Selected Recommendations"):
            if not selected_recommendations:
                st.warning("Select at least one recommendation.")
            else:
                updated_dataframe = apply_ai_recommended_features(
                    current_feature_dataset,
                    selected_recommendations,
                )
                apply_feature_engineering_result(updated_dataframe, "AI Recommended Features")

elif selected_operation == "Interaction Features":
    st.subheader("Interaction Feature Generator")
    if len(numerical_columns) < 2:
        st.warning("At least two numerical columns are required.")
    else:
        left_col, right_col = st.columns(2)
        with left_col:
            column_1 = st.selectbox("Select First Feature", options=numerical_columns)
        with right_col:
            column_2 = st.selectbox(
                "Select Second Feature",
                options=numerical_columns,
                index=1,
            )
        operation = st.selectbox(
            "Select Operation",
            options=["add", "subtract", "multiply", "divide"],
        )
        if st.button("Generate Interaction Feature"):
            updated_dataframe = create_interaction_feature(
                current_feature_dataset,
                column_1,
                column_2,
                operation,
            )
            apply_feature_engineering_result(updated_dataframe, "Interaction Features")

elif selected_operation == "Polynomial Features":
    st.subheader("Polynomial Feature Generator")
    if not numerical_columns:
        st.warning("No numerical columns detected.")
    else:
        selected_columns = st.multiselect("Select Numerical Features", options=numerical_columns)
        degree = st.slider("Polynomial Degree", min_value=2, max_value=4, value=2)
        if st.button("Generate Polynomial Features"):
            if not selected_columns:
                st.warning("Please select at least one feature.")
            else:
                updated_dataframe = create_polynomial_features(
                    current_feature_dataset,
                    selected_columns,
                    degree,
                )
                apply_feature_engineering_result(updated_dataframe, "Polynomial Features")

elif selected_operation == "Log Transformation":
    st.subheader("Log Transformation")
    if not numerical_columns:
        st.warning("No numerical columns detected.")
    else:
        selected_column = st.selectbox("Select Numerical Column", options=numerical_columns)
        if st.button("Apply Log Transformation"):
            updated_dataframe = apply_log_transformation(current_feature_dataset, selected_column)
            apply_feature_engineering_result(updated_dataframe, "Log Transformation")

elif selected_operation == "Feature Binning":
    st.subheader("Feature Binning")
    if not numerical_columns:
        st.warning("No numerical columns detected.")
    else:
        selected_column = st.selectbox("Select Numerical Column", options=numerical_columns)
        bins = st.slider("Number of Bins", min_value=2, max_value=20, value=5)
        if st.button("Apply Feature Binning"):
            updated_dataframe = apply_feature_binning(
                current_feature_dataset,
                selected_column,
                bins,
            )
            apply_feature_engineering_result(updated_dataframe, "Feature Binning")

elif selected_operation == "Datetime Features":
    st.subheader("Datetime Feature Extraction")
    if not datetime_columns:
        st.warning("No datetime columns detected.")
    else:
        selected_datetime_column = st.selectbox("Select Datetime Column", options=datetime_columns)
        if st.button("Extract Datetime Features"):
            updated_dataframe = extract_datetime_features(current_feature_dataset, selected_datetime_column)
            apply_feature_engineering_result(updated_dataframe, "Datetime Features")

st.markdown("---")
st.subheader("Engineered Dataset Preview")
preview_rows = st.slider(
    "Preview Rows",
    min_value=5,
    max_value=min(50, max(len(current_feature_dataset), 5)),
    value=min(10, max(len(current_feature_dataset), 5)),
)
st.dataframe(current_feature_dataset.head(preview_rows), width="stretch")

if target_column in current_feature_dataset.columns:
    new_feature_columns = [
        column
        for column in current_feature_dataset.columns
        if column != target_column and column not in source_feature_dataset.columns
    ]
    if new_feature_columns:
        st.markdown("---")
        st.subheader("New Feature Columns")
        st.dataframe(pd.DataFrame({"Feature": new_feature_columns}), width="stretch")

st.markdown("---")
render_engineered_data_section(session_state=st.session_state, expanded=True)
st.markdown("---")
st.success("Feature engineering module ready.")
if st.session_state.preprocessing_completed and st.session_state.preprocessing_results is not None:
    st.info("Next Step -> Navigate to 'Model Training' page.")
else:
    st.info("Next Step -> Navigate back to 'Preprocessing' to configure the new columns.")
