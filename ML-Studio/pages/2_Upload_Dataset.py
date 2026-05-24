from io import BytesIO

import pandas as pd
import streamlit as st

from src.data_loader import (
    MAX_UPLOAD_SIZE_MB,
    SUPPORTED_UPLOAD_TYPES,
    detect_target_candidates,
    get_dataset_summary,
    load_dataset,
    validate_uploaded_file,
)
from src.model_selector import detect_problem_type
from src.session_manager import initialize_session_state, set_dataset, set_target_column
from src.ui_helpers import (
    load_ml_studio_css,
    render_engineered_data_section,
    render_metric_row,
    render_page_header,
)


st.set_page_config(page_title="Upload Dataset | ML Studio", page_icon=":material/upload_file:", layout="wide")
load_ml_studio_css()
initialize_session_state(st.session_state)

render_page_header(
    "Upload Dataset",
    "Load a CSV, Excel, or JSON dataset and choose the target column for the workflow.",
    "Data Intake",
)


@st.cache_data(show_spinner=False)
def _load_dataset_from_bytes(raw_bytes: bytes, file_type: str) -> pd.DataFrame:
    return load_dataset(BytesIO(raw_bytes), file_type)


with st.sidebar:
    st.markdown("### Upload Rules")
    st.info(f"Maximum file size: {MAX_UPLOAD_SIZE_MB} MB")
    st.markdown("### Supported Formats")
    st.write("- CSV (.csv)")
    st.write("- Excel (.xlsx, .xls)")
    st.write("- JSON (.json)")
    st.markdown("### Validation")
    st.write("- Non-empty dataset")
    st.write("- Unique column names")
    st.write("- At least one usable feature column")

with st.container(border=True):
    control_col, upload_col = st.columns([1.1, 2.2], gap="large")
    with control_col:
        selected_file_type = st.selectbox(
            "Dataset Format",
            options=list(SUPPORTED_UPLOAD_TYPES.keys()),
            format_func=lambda value: SUPPORTED_UPLOAD_TYPES[value]["label"],
        )
        st.caption("Choose the format before selecting a file so validation matches the upload.")
    with upload_col:
        uploaded_file = st.file_uploader(
            "Upload Dataset File",
            type=SUPPORTED_UPLOAD_TYPES[selected_file_type]["streamlit_types"],
            key=f"dataset_upload_{selected_file_type}",
        )

if uploaded_file is not None:
    is_valid, validation_message = validate_uploaded_file(uploaded_file, selected_file_type)
    if not is_valid:
        if "upload limit" in validation_message.lower():
            st.warning(validation_message)
        else:
            st.error(validation_message)
        st.stop()

    try:
        dataframe = _load_dataset_from_bytes(uploaded_file.getvalue(), selected_file_type)
        dataset_changed = set_dataset(st.session_state, dataframe, uploaded_file.name)
        if dataset_changed:
            st.success("Dataset uploaded and downstream artifacts were reset.")
        else:
            st.success("Dataset already loaded.")
    except Exception as error:
        st.error(f"Error while loading dataset: {error}")
        st.stop()

dataset = st.session_state.dataset

if dataset is None:
    st.info("Upload a dataset to begin analysis.")
    st.stop()

summary = get_dataset_summary(dataset)

st.markdown("---")
st.subheader("Dataset Overview")
with st.container(border=True):
    render_metric_row(
        [
            ("Rows", summary["rows"]),
            ("Columns", summary["columns"]),
            ("Missing Values", summary["missing_values"]),
            ("Duplicate Rows", summary["duplicate_rows"]),
        ],
        columns=4,
    )

st.markdown("---")
preview_col, schema_col = st.columns([1.45, 1.0], gap="large")
with preview_col:
    st.subheader("Dataset Preview")
    preview_rows = st.slider("Number of preview rows", min_value=5, max_value=50, value=10)
    st.dataframe(dataset.head(preview_rows), width="stretch")

with schema_col:
    st.subheader("Column Information")
    st.dataframe(
        pd.DataFrame(
            {
                "Column": dataset.columns,
                "Data Type": dataset.dtypes.astype(str),
                "Missing Values": dataset.isna().sum().values,
                "Unique Values": dataset.nunique(dropna=True).values,
            }
        ),
        width="stretch",
        hide_index=True,
    )

st.markdown("---")
target_col, problem_col = st.columns([1.1, 1.0], gap="large")
with target_col:
    with st.container(border=True):
        st.subheader("Target Column Selection")
        target_candidates = detect_target_candidates(dataset)
        default_target = (
            st.session_state.target_column
            if st.session_state.target_column in dataset.columns
            else (target_candidates[0] if target_candidates else dataset.columns[0])
        )
        target_column = st.selectbox(
            "Select Target Column",
            options=list(dataset.columns),
            index=list(dataset.columns).index(default_target),
        )
        target_changed = set_target_column(st.session_state, target_column)
        if target_changed:
            st.info("Target changed; stale preprocessing and model artifacts were cleared.")

        target_series = dataset[target_column]
        render_metric_row(
            [
                ("Unique Values", target_series.nunique(dropna=True)),
                ("Missing Values", int(target_series.isna().sum())),
                ("Data Type", str(target_series.dtype)),
            ],
            columns=3,
        )

with problem_col:
    with st.container(border=True):
        st.subheader("Problem Type Detection")
        try:
            detected_problem_type = detect_problem_type(dataset[target_column])
            st.session_state.problem_type = detected_problem_type
            st.success(f"{detected_problem_type.title()} problem detected")
        except Exception as error:
            st.error(f"Problem type detection failed: {error}")

st.markdown("---")
st.info("Next step: Dataset Overview")
render_engineered_data_section(session_state=st.session_state)
