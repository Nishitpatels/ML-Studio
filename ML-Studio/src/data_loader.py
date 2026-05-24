"""Dataset loading and validation helpers."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.api.types import is_numeric_dtype


MAX_UPLOAD_SIZE_MB = 50
SUPPORTED_UPLOAD_TYPES = {
    "csv": {
        "label": "CSV",
        "extensions": [".csv"],
        "streamlit_types": ["csv"],
    },
    "excel": {
        "label": "Excel",
        "extensions": [".xlsx", ".xls"],
        "streamlit_types": ["xlsx", "xls"],
    },
    "json": {
        "label": "JSON",
        "extensions": [".json"],
        "streamlit_types": ["json"],
    },
}


def _get_file_size(uploaded_file: Any) -> int | None:
    size = getattr(uploaded_file, "size", None)
    if size is not None:
        return int(size)
    if hasattr(uploaded_file, "getbuffer"):
        return len(uploaded_file.getbuffer())
    return None


def validate_uploaded_file(uploaded_file: Any, file_type: str = "csv") -> tuple[bool, str]:
    """Validate an uploaded dataset before parsing."""

    if uploaded_file is None:
        return False, "No file uploaded."

    if file_type not in SUPPORTED_UPLOAD_TYPES:
        return False, "Unsupported upload type selected."

    upload_type = SUPPORTED_UPLOAD_TYPES[file_type]
    file_name = getattr(uploaded_file, "name", "")
    file_extension = Path(file_name).suffix.lower()
    if file_extension not in upload_type["extensions"]:
        expected = ", ".join(upload_type["extensions"])
        return False, f"Selected {upload_type['label']} upload expects: {expected}."

    file_size = _get_file_size(uploaded_file)
    if file_size == 0:
        return False, "Uploaded file is empty."
    if file_size and file_size > MAX_UPLOAD_SIZE_MB * 1024 * 1024:
        uploaded_size_mb = round(file_size / (1024 * 1024), 2)
        return (
            False,
            f"File is {uploaded_size_mb} MB, which exceeds the {MAX_UPLOAD_SIZE_MB} MB upload limit. "
            "Please upload a smaller dataset file.",
        )

    return True, "Upload is valid."


def _validate_dataframe(dataframe: pd.DataFrame, format_name: str) -> pd.DataFrame:
    """Validate a parsed dataframe while preserving the existing workflow contract."""

    if dataframe.empty:
        raise ValueError(f"{format_name} file contains no data rows.")
    if dataframe.shape[1] == 0:
        raise ValueError(f"{format_name} file contains no columns.")
    if dataframe.columns.duplicated().any():
        duplicates = dataframe.columns[dataframe.columns.duplicated()].tolist()
        raise ValueError(f"Duplicate column names detected: {duplicates}")
    if all(dataframe[column].isna().all() for column in dataframe.columns):
        raise ValueError("All uploaded columns are empty.")
    return dataframe


def load_csv(uploaded_file: Any) -> pd.DataFrame:
    """Read a CSV file-like object into a validated dataframe."""

    if hasattr(uploaded_file, "getvalue"):
        raw_bytes = uploaded_file.getvalue()
        buffer = BytesIO(raw_bytes)
    else:
        uploaded_file.seek(0)
        buffer = uploaded_file

    dataframe = pd.read_csv(buffer)
    return _validate_dataframe(dataframe, "CSV")


def load_excel(uploaded_file: Any) -> pd.DataFrame:
    """Read an Excel file-like object into a validated dataframe."""

    if hasattr(uploaded_file, "getvalue"):
        raw_bytes = uploaded_file.getvalue()
        buffer = BytesIO(raw_bytes)
    else:
        uploaded_file.seek(0)
        buffer = uploaded_file

    try:
        dataframe = pd.read_excel(buffer)
    except ImportError as error:
        raise ImportError(
            "Excel uploads require openpyxl for .xlsx files or xlrd for .xls files. "
            "Install project requirements and restart Streamlit."
        ) from error

    return _validate_dataframe(dataframe, "Excel")


def load_json(uploaded_file: Any) -> pd.DataFrame:
    """Read a JSON file-like object into a validated dataframe."""

    if hasattr(uploaded_file, "getvalue"):
        raw_bytes = uploaded_file.getvalue()
    else:
        uploaded_file.seek(0)
        raw_bytes = uploaded_file.read()

    try:
        dataframe = pd.read_json(BytesIO(raw_bytes))
    except ValueError:
        dataframe = pd.read_json(BytesIO(raw_bytes), lines=True)

    return _validate_dataframe(dataframe, "JSON")


def load_dataset(uploaded_file: Any, file_type: str) -> pd.DataFrame:
    """Load a supported dataset file into a dataframe."""

    if file_type == "csv":
        return load_csv(uploaded_file)
    if file_type == "excel":
        return load_excel(uploaded_file)
    if file_type == "json":
        return load_json(uploaded_file)
    raise ValueError("Unsupported upload type selected.")


def get_dataset_summary(dataframe: pd.DataFrame) -> dict[str, int]:
    return {
        "rows": int(dataframe.shape[0]),
        "columns": int(dataframe.shape[1]),
        "missing_values": int(dataframe.isna().sum().sum()),
        "duplicate_rows": int(dataframe.duplicated().sum()),
    }


def detect_target_candidates(dataframe: pd.DataFrame) -> list[str]:
    """Rank plausible target columns while avoiding obvious identifiers."""

    scored_columns: list[tuple[int, str]] = []
    row_count = max(len(dataframe), 1)

    for column in dataframe.columns:
        lowered = column.lower()
        if lowered == "id" or lowered.endswith("_id") or "index" in lowered:
            continue

        unique_count = dataframe[column].nunique(dropna=True)
        unique_ratio = unique_count / row_count
        score = 0

        if lowered in {"target", "label", "outcome", "class", "y"}:
            score += 100
        if unique_count <= 20:
            score += 20
        elif unique_ratio < 0.2:
            score += 10
        if not is_numeric_dtype(dataframe[column]):
            score += 5

        scored_columns.append((score, column))

    scored_columns.sort(key=lambda item: (-item[0], dataframe.columns.get_loc(item[1])))
    return [column for _, column in scored_columns]


def get_numerical_columns(dataframe: pd.DataFrame) -> list[str]:
    return dataframe.select_dtypes(include="number").columns.tolist()


def get_categorical_columns(dataframe: pd.DataFrame) -> list[str]:
    return dataframe.select_dtypes(exclude="number").columns.tolist()


def get_datetime_columns(dataframe: pd.DataFrame) -> list[str]:
    datetime_columns: list[str] = []
    for column in dataframe.columns:
        if pd.api.types.is_datetime64_any_dtype(dataframe[column]):
            datetime_columns.append(column)
            continue
        if dataframe[column].dtype == "object":
            parsed = pd.to_datetime(dataframe[column], errors="coerce")
            if parsed.notna().mean() >= 0.8 and parsed.nunique(dropna=True) > 1:
                datetime_columns.append(column)
    return datetime_columns


def calculate_dataset_health_score(dataframe: pd.DataFrame) -> dict[str, Any]:
    score = 100
    warnings: list[str] = []
    missing_ratio = dataframe.isna().sum().sum() / max(dataframe.size, 1)
    duplicate_ratio = dataframe.duplicated().sum() / max(len(dataframe), 1)

    if missing_ratio > 0.3:
        score -= 25
        warnings.append("High missing-value ratio detected.")
    if duplicate_ratio > 0.1:
        score -= 15
        warnings.append("High duplicate-row ratio detected.")

    for column in dataframe.columns:
        if dataframe[column].nunique(dropna=True) / max(len(dataframe), 1) > 0.9:
            warnings.append(f"Column '{column}' has very high cardinality.")

    return {"health_score": max(score, 0), "warnings": warnings}


def get_random_sample(dataframe: pd.DataFrame, sample_size: int = 5) -> pd.DataFrame:
    return dataframe if len(dataframe) <= sample_size else dataframe.sample(sample_size, random_state=42)


def reset_uploaded_file_pointer(uploaded_file: Any) -> None:
    if hasattr(uploaded_file, "seek"):
        uploaded_file.seek(0)
