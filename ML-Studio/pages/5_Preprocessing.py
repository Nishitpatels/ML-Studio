import pandas as pd
import streamlit as st

from src.preprocessing import (
    execute_preprocessing_pipeline,
    generate_preprocessing_recommendations,
    generate_preprocessing_summary,
    get_default_column_settings,
)
from src.session_manager import (
    apply_workflow_stage_result,
    get_workflow_stage_dataset,
    initialize_session_state,
    reset_workflow_stage,
    set_preprocessing_results,
    set_working_dataset,
    sync_workflow_stage,
    undo_workflow_stage,
)
from src.ui_helpers import load_ml_studio_css, render_engineered_data_section, render_metric_row, render_page_header


st.set_page_config(
    page_title="Preprocessing | ML Studio",
    page_icon=":material/tune:",
    layout="wide",
)

load_ml_studio_css()
render_page_header(
    "Dynamic Preprocessing Engine",
    "Clean, configure, and rerun preprocessing without breaking the workflow.",
    "Preprocessing",
)
initialize_session_state(st.session_state)


def _format_option(value: str) -> str:
    mapping = {
        "none": "None",
        "mean": "Mean",
        "median": "Median",
        "most_frequent": "Most Frequent",
        "constant": "Constant",
        "onehot": "One-Hot",
        "ordinal": "Ordinal",
        "standard": "StandardScaler",
        "minmax": "MinMaxScaler",
        "robust": "RobustScaler",
    }
    return mapping.get(value, value.replace("_", " ").title())


def render_preprocessing_results(preprocessing_results, target_column: str):
    cleaning_summary = preprocessing_results.get("cleaning_summary", {})
    transformed_dataframe = preprocessing_results.get("transformed_dataframe")

    st.markdown("---")
    with st.container(border=True):
        st.subheader("Processed Dataset Summary")
        render_metric_row(
            [
                ("Rows Remaining", cleaning_summary.get("remaining_rows", 0)),
                ("Rows Dropped", cleaning_summary.get("dropped_rows", 0)),
                ("Columns Remaining", cleaning_summary.get("remaining_columns", 0)),
                ("Columns Dropped", cleaning_summary.get("dropped_columns", 0)),
            ],
            columns=4,
        )
        render_metric_row(
            [
                ("Duplicate Rows Before", cleaning_summary.get("duplicate_rows_before", 0)),
                ("Duplicate Rows Removed", cleaning_summary.get("duplicate_rows_removed", 0)),
            ],
            columns=2,
        )

    details = []
    if cleaning_summary.get("manually_dropped_columns"):
        details.append("Manual column drops: " + ", ".join(cleaning_summary["manually_dropped_columns"]))
    if cleaning_summary.get("columns_dropped_for_missing_values"):
        details.append(
            "Columns removed for missing values: "
            + ", ".join(cleaning_summary["columns_dropped_for_missing_values"])
        )
    if cleaning_summary.get("internally_dropped_columns"):
        details.append(
            "Columns dropped from modeling: " + ", ".join(cleaning_summary["internally_dropped_columns"])
        )
    if cleaning_summary.get("drop_rows_with_missing"):
        threshold_percent = int((cleaning_summary.get("row_non_null_threshold") or 0) * 100)
        details.append(f"Row retention threshold applied: {threshold_percent}% non-null values required.")
    details.append(
        "Duplicate handling: "
        + ("Remove duplicates" if cleaning_summary.get("duplicate_handling") == "remove" else "Keep duplicates")
    )
    with st.container(border=True):
        st.subheader("Cleaning Decisions")
        for item in details:
            st.info(item)

    st.markdown("---")
    with st.container(border=True):
        st.subheader("Training Split Status")
        render_metric_row(
            [
                ("Training Samples", preprocessing_results["X_train"].shape[0]),
                ("Testing Samples", preprocessing_results["X_test"].shape[0]),
                ("Processed Features", preprocessing_results["X_train"].shape[1]),
                ("Target Column", target_column),
            ],
            columns=4,
        )

    st.markdown("---")
    st.subheader("Transformed DataFrame Preview")
    preview_rows = st.slider(
        "Preview Rows",
        min_value=5,
        max_value=min(50, max(len(transformed_dataframe), 5)),
        value=min(10, max(len(transformed_dataframe), 5)),
        key="preprocessing_preview_rows",
    )
    st.dataframe(transformed_dataframe.head(preview_rows), width="stretch")

    st.markdown("---")
    st.subheader("Processed Feature Columns")
    st.dataframe(pd.DataFrame({"Feature": preprocessing_results["X_train"].columns}), width="stretch")

    column_assessments = preprocessing_results.get("column_assessments", [])
    if column_assessments:
        assessment_frame = pd.DataFrame(column_assessments).rename(
            columns={
                "auto_drop": "suggested_manual_drop",
            }
        )
        st.markdown("---")
        st.subheader("Guided Column Review")
        st.dataframe(assessment_frame, width="stretch")


def restore_preprocessing_state(
    restored_dataset: pd.DataFrame,
    restored_payload: dict | None,
    default_source_stage: str,
):
    source_stage = (restored_payload or {}).get("source_stage", default_source_stage)
    is_feature_engineering_source = source_stage == "feature_engineering"

    if restored_payload and restored_payload.get("preprocessing_results") is not None:
        set_working_dataset(
            st.session_state,
            restored_dataset,
            engineered=False,
            preserve_engineered_dataset=is_feature_engineering_source,
        )
        set_preprocessing_results(
            st.session_state,
            restored_payload["preprocessing_results"],
            restored_payload.get("preprocessing_config"),
        )
        if is_feature_engineering_source:
            st.session_state.engineered_dataset = get_workflow_stage_dataset(
                st.session_state,
                "preprocessing",
                snapshot_name="previous_stage_dataset",
            )
        return

    set_working_dataset(
        st.session_state,
        restored_dataset,
        engineered=is_feature_engineering_source,
    )
    if is_feature_engineering_source:
        st.session_state.engineered_dataset = restored_dataset.copy()


def render_column_controls(dataset: pd.DataFrame, target_column: str, default_settings: dict[str, dict]):
    updated_settings: dict[str, dict] = {}
    feature_columns = [column for column in dataset.columns if column != target_column]

    grouped_columns = {
        "Numerical": [column for column in feature_columns if default_settings[column]["column_type"] == "numerical"],
        "Categorical": [column for column in feature_columns if default_settings[column]["column_type"] == "categorical"],
        "Datetime": [column for column in feature_columns if default_settings[column]["column_type"] == "datetime"],
    }

    tabs = st.tabs(["Numerical", "Categorical", "Datetime"])
    for tab, (group_name, columns) in zip(tabs, grouped_columns.items()):
        with tab:
            if not columns:
                st.info(f"No {group_name.lower()} columns detected.")
                continue

            for column_name in columns:
                settings = default_settings[column_name]
                missing_ratio = dataset[column_name].isna().mean()
                data_type = str(dataset[column_name].dtype)

                with st.container(border=True):
                    header_col1, header_col2, header_col3 = st.columns([2.4, 1.2, 1.0])
                    header_col1.markdown(f"**{column_name}**")
                    header_col2.caption(f"Datatype: {data_type}")
                    header_col3.caption(f"Missing: {missing_ratio:.1%}")

                    if settings["column_type"] == "datetime":
                        control_col1, control_col2, control_col3, control_col4 = st.columns([1.4, 1.4, 1.0, 0.8])
                        imputation = control_col1.selectbox(
                            "Imputation",
                            options=["none", "mean", "median", "most_frequent"],
                            index=["none", "mean", "median", "most_frequent"].index(
                                settings.get("imputation", "median")
                            ),
                            key=f"preprocessing_imputation_{column_name}",
                            format_func=_format_option,
                        )
                        scaling = control_col2.selectbox(
                            "Scaling",
                            options=["none", "standard", "minmax", "robust"],
                            index=["none", "standard", "minmax", "robust"].index(
                                settings.get("scaling", "standard")
                            ),
                            key=f"preprocessing_scaling_{column_name}",
                            format_func=_format_option,
                        )
                        control_col3.caption("Encoding is not needed for datetime extraction.")
                        drop_value = control_col4.checkbox(
                            "Drop",
                            value=bool(settings.get("drop", False)),
                            key=f"preprocessing_drop_{column_name}",
                        )
                        updated_settings[column_name] = {
                            "column_type": settings["column_type"],
                            "drop": drop_value,
                            "imputation": imputation,
                            "scaling": scaling,
                        }
                        continue

                    control_col1, control_col2, control_col3, control_col4 = st.columns([1.2, 1.2, 1.2, 0.8])
                    drop_value = control_col4.checkbox(
                        "Drop",
                        value=bool(settings.get("drop", False)),
                        key=f"preprocessing_drop_{column_name}",
                    )

                    if settings["column_type"] == "categorical":
                        imputation = control_col1.selectbox(
                            "Imputation",
                            options=["none", "most_frequent", "constant"],
                            index=["none", "most_frequent", "constant"].index(
                                settings.get("imputation", "most_frequent")
                            ),
                            key=f"preprocessing_imputation_{column_name}",
                            format_func=_format_option,
                            disabled=drop_value,
                        )
                        encoding = control_col2.selectbox(
                            "Encoding",
                            options=["none", "ordinal", "onehot"],
                            index=["none", "ordinal", "onehot"].index(settings.get("encoding", "onehot")),
                            key=f"preprocessing_encoding_{column_name}",
                            format_func=_format_option,
                            disabled=drop_value,
                        )
                        scaling = control_col3.selectbox(
                            "Scaling",
                            options=["none", "standard", "minmax", "robust"],
                            index=["none", "standard", "minmax", "robust"].index(
                                settings.get("scaling", "none")
                            ),
                            key=f"preprocessing_scaling_{column_name}",
                            format_func=_format_option,
                            disabled=drop_value or encoding == "none",
                        )
                        if encoding == "none":
                            scaling = "none"
                        updated_settings[column_name] = {
                            "column_type": settings["column_type"],
                            "drop": drop_value,
                            "imputation": imputation,
                            "encoding": encoding,
                            "scaling": scaling,
                        }
                    else:
                        imputation = control_col1.selectbox(
                            "Imputation",
                            options=["none", "mean", "median", "most_frequent"],
                            index=["none", "mean", "median", "most_frequent"].index(
                                settings.get("imputation", "median")
                            ),
                            key=f"preprocessing_imputation_{column_name}",
                            format_func=_format_option,
                            disabled=drop_value,
                        )
                        encoding = control_col2.selectbox(
                            "Encoding",
                            options=["none", "ordinal", "onehot"],
                            index=["none", "ordinal", "onehot"].index(settings.get("encoding", "none")),
                            key=f"preprocessing_encoding_{column_name}",
                            format_func=_format_option,
                            disabled=drop_value,
                        )
                        scaling = control_col3.selectbox(
                            "Scaling",
                            options=["none", "standard", "minmax", "robust"],
                            index=["none", "standard", "minmax", "robust"].index(
                                settings.get("scaling", "standard")
                            ),
                            key=f"preprocessing_scaling_{column_name}",
                            format_func=_format_option,
                            disabled=drop_value,
                        )
                        updated_settings[column_name] = {
                            "column_type": settings["column_type"],
                            "drop": drop_value,
                            "imputation": imputation,
                            "encoding": encoding,
                            "scaling": scaling,
                        }

    return updated_settings


source_dataset = (
    st.session_state.engineered_dataset.copy()
    if st.session_state.engineered_dataset is not None
    else st.session_state.working_dataset.copy()
    if st.session_state.working_dataset is not None
    else st.session_state.dataset.copy()
    if st.session_state.dataset is not None
    else None
)
target_column = st.session_state.target_column

if source_dataset is None:
    st.warning("No dataset found. Please upload a dataset first.")
    st.stop()

if target_column is None:
    st.warning("Please select a target column first.")
    st.stop()

stage_state = sync_workflow_stage(st.session_state, "preprocessing", source_dataset)
page_input_dataset = get_workflow_stage_dataset(
    st.session_state,
    "preprocessing",
    snapshot_name="previous_stage_dataset",
)
current_stage_dataset = get_workflow_stage_dataset(st.session_state, "preprocessing")
stage_payload = stage_state.get("current_payload") or {}
current_config = stage_payload.get("preprocessing_config") or st.session_state.get("preprocessing_config") or {}
source_stage = (
    "feature_engineering"
    if st.session_state.engineered_dataset is not None
    else "preprocessing"
    if st.session_state.preprocessing_completed
    else "eda"
)

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
                ("Previous Stage Snapshot", f"{page_input_dataset.shape[0]} x {page_input_dataset.shape[1]}"),
                ("Current Page Snapshot", f"{current_stage_dataset.shape[0]} x {current_stage_dataset.shape[1]}"),
            ],
            columns=3,
        )
with top_col2:
    with st.container(border=True):
        st.subheader("Page Controls")
        undo_col, reset_col = st.columns(2)
        if undo_col.button("Undo Preprocessing", width="stretch"):
            restored_dataset, restored_payload = undo_workflow_stage(st.session_state, "preprocessing")
            restore_preprocessing_state(restored_dataset, restored_payload, source_stage)
            st.success("Previous preprocessing state restored.")
            st.rerun()
        if reset_col.button("Reset Current Page Changes", width="stretch"):
            restored_dataset, restored_payload = reset_workflow_stage(st.session_state, "preprocessing")
            restore_preprocessing_state(restored_dataset, restored_payload, source_stage)
            st.success("Preprocessing page reset to its incoming dataset.")
            st.rerun()

duplicate_count = int(page_input_dataset.duplicated().sum())

with st.sidebar:
    st.markdown("## Preprocessing Settings")
    st.caption("Column-level choices stay manual so the workflow remains predictable.")

    drop_rows_with_missing = st.checkbox(
        "Drop Rows With Missing Values",
        value=current_config.get("drop_rows_with_missing", False),
        help="Drop incomplete rows before feature-level preprocessing.",
    )
    row_non_null_threshold = st.slider(
        "Minimum Non-Null Row Ratio",
        min_value=0.5,
        max_value=1.0,
        value=float(current_config.get("row_non_null_threshold", 1.0)),
        step=0.05,
        disabled=not drop_rows_with_missing,
        help="Rows below this non-null ratio will be dropped.",
    )
    drop_columns_with_missing = st.checkbox(
        "Drop Columns With Missing Values",
        value=current_config.get("drop_columns_with_missing", False),
        help="Remove feature columns that still contain missing values before transformation.",
    )
    duplicate_handling = st.radio(
        "Duplicate Row Handling",
        options=["keep", "remove"],
        index=0 if current_config.get("duplicate_handling", "keep") == "keep" else 1,
        format_func=lambda value: "Keep Duplicates" if value == "keep" else "Remove Duplicates",
        help="Choose whether exact duplicate rows should remain in the dataset before training.",
    )
    show_duplicate_preview = st.checkbox(
        "Show Duplicate Preview",
        value=current_config.get("show_duplicate_preview", False),
        disabled=duplicate_count == 0,
    )
    test_size = st.slider(
        "Test Size",
        min_value=0.1,
        max_value=0.4,
        value=float(current_config.get("test_size", 0.2)),
        step=0.05,
    )

st.markdown("---")
with st.container(border=True):
    st.subheader("Dataset Information")
    render_metric_row(
        [
            ("Rows", page_input_dataset.shape[0]),
            ("Columns", page_input_dataset.shape[1]),
            ("Target Column", target_column),
            (
                "Source Stage",
                "Feature Engineering" if source_stage == "feature_engineering" else "Workflow Dataset",
            ),
        ],
        columns=4,
    )
    render_metric_row(
        [
            ("Duplicate Count", duplicate_count),
            ("Duplicate Percentage", f"{(duplicate_count / max(len(page_input_dataset), 1)) * 100:.2f}%"),
        ],
        columns=2,
    )

if show_duplicate_preview and duplicate_count > 0:
    st.markdown("---")
    st.subheader("Duplicate Preview")
    duplicate_preview = page_input_dataset[page_input_dataset.duplicated(keep=False)].head(50)
    st.dataframe(duplicate_preview, width="stretch")

st.markdown("---")
st.subheader("ML Studio Recommendations")
recommendations = generate_preprocessing_recommendations(page_input_dataset, target_column)
if recommendations:
    for recommendation in recommendations:
        st.info(recommendation)
else:
    st.success("Dataset preprocessing requirements look balanced.")

st.markdown("---")
st.subheader("Column-wise Preprocessing Controls")
st.caption("Each row shows the column name, datatype, missing percentage, and the controls used in the pipeline.")
column_settings = render_column_controls(
    page_input_dataset,
    target_column,
    get_default_column_settings(
        page_input_dataset,
        target_column,
        current_config.get("column_settings"),
    ),
)
columns_to_drop = [column for column, settings in column_settings.items() if settings.get("drop", False)]

st.markdown("---")
st.subheader("Selected Preprocessing Configuration")
configuration_summary = generate_preprocessing_summary(
    column_settings=column_settings,
    drop_rows_with_missing=drop_rows_with_missing,
    row_non_null_threshold=row_non_null_threshold,
    drop_columns_with_missing=drop_columns_with_missing,
    columns_to_drop=columns_to_drop,
    duplicate_handling=duplicate_handling,
)
st.dataframe(
    pd.DataFrame(
        {
            "Configuration": configuration_summary.keys(),
            "Selected Option": configuration_summary.values(),
        }
    ),
    width="stretch",
)

st.markdown("---")
execute_preprocessing = st.button("Execute Preprocessing Pipeline")

results_to_display = None
if execute_preprocessing:
    with st.spinner("Generating preprocessing pipeline..."):
        try:
            preprocessing_results = execute_preprocessing_pipeline(
                dataframe=page_input_dataset,
                target_column=target_column,
                column_settings=column_settings,
                test_size=test_size,
                columns_to_drop=columns_to_drop,
                drop_rows_with_missing=drop_rows_with_missing,
                row_non_null_threshold=row_non_null_threshold,
                drop_columns_with_missing=drop_columns_with_missing,
                duplicate_handling=duplicate_handling,
            )

            preprocessing_config = {
                "test_size": test_size,
                "columns_to_drop": columns_to_drop,
                "drop_rows_with_missing": drop_rows_with_missing,
                "row_non_null_threshold": row_non_null_threshold,
                "drop_columns_with_missing": drop_columns_with_missing,
                "duplicate_handling": duplicate_handling,
                "show_duplicate_preview": show_duplicate_preview,
                "column_settings": column_settings,
            }

            set_working_dataset(
                st.session_state,
                preprocessing_results["cleaned_dataframe"],
                engineered=False,
                preserve_engineered_dataset=source_stage == "feature_engineering",
            )
            if source_stage == "feature_engineering":
                st.session_state.engineered_dataset = page_input_dataset.copy()

            set_preprocessing_results(
                st.session_state,
                preprocessing_results,
                preprocessing_config,
            )
            apply_workflow_stage_result(
                st.session_state,
                "preprocessing",
                preprocessing_results["cleaned_dataframe"],
                payload={
                    "preprocessing_results": preprocessing_results,
                    "preprocessing_config": preprocessing_config,
                    "source_stage": source_stage,
                },
            )

            st.success("Preprocessing pipeline executed successfully.")
            results_to_display = preprocessing_results
        except Exception as error:
            st.error(f"Preprocessing failed: {error}")

if results_to_display is None and st.session_state.preprocessing_completed and st.session_state.preprocessing_results is not None:
    st.markdown("---")
    st.success("Preprocessing pipeline is available in the current session.")
    results_to_display = st.session_state.preprocessing_results

if results_to_display is not None:
    render_preprocessing_results(results_to_display, target_column)

render_engineered_data_section(session_state=st.session_state)
st.markdown("---")
st.info("Next Step -> Navigate to 'Feature Engineering' or continue to 'Model Training'.")
