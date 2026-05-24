"""Preprocessing utilities and schema-aware transformers."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, OneHotEncoder, OrdinalEncoder, RobustScaler, StandardScaler

from src.analyzer import assess_duplicate_row_recommendation
from src.model_selector import detect_problem_type


class FeatureSchemaAligner(BaseEstimator, TransformerMixin):
    """Align inference frames to the feature schema learned at training time."""

    def fit(self, X: pd.DataFrame, y: Any = None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        self.feature_names_in_ = list(X.columns)
        self.dtypes_ = {column: str(dtype) for column, dtype in X.dtypes.items()}
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names_in_)

        aligned = X.copy()
        for column in self.feature_names_in_:
            if column not in aligned.columns:
                aligned[column] = np.nan

        return aligned[self.feature_names_in_]

    def get_feature_names_out(self, input_features=None):
        columns = self.feature_names_in_ if input_features is None else list(input_features)
        return np.asarray(columns, dtype=object)


class CategoricalCaster(BaseEstimator, TransformerMixin):
    """Normalize mixed categorical columns so sklearn imputers remain stable."""

    def fit(self, X: pd.DataFrame, y: Any = None):
        self.feature_names_in_ = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        dataframe = pd.DataFrame(X, columns=self.feature_names_in_).copy()
        dataframe = dataframe.astype("object")
        dataframe[pd.isna(dataframe)] = np.nan
        return dataframe

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_in_ if input_features is None else input_features, dtype=object)


class DateTimeFeatureExtractor(BaseEstimator, TransformerMixin):
    """Expand datetime columns into numeric calendar features."""

    def fit(self, X: pd.DataFrame, y: Any = None):
        self.feature_names_in_ = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        dataframe = pd.DataFrame(X, columns=self.feature_names_in_)
        features: dict[str, pd.Series] = {}
        for column in self.feature_names_in_:
            parsed = pd.to_datetime(dataframe[column], errors="coerce")
            features[f"{column}_year"] = parsed.dt.year
            features[f"{column}_month"] = parsed.dt.month
            features[f"{column}_day"] = parsed.dt.day
            features[f"{column}_dayofweek"] = parsed.dt.dayofweek
        return pd.DataFrame(features, index=dataframe.index)

    def get_feature_names_out(self, input_features=None):
        columns = self.feature_names_in_ if input_features is None else list(input_features)
        return np.asarray(
            [
                f"{column}_{suffix}"
                for column in columns
                for suffix in ("year", "month", "day", "dayofweek")
            ],
            dtype=object,
        )


class ColumnSubsetSelector(BaseEstimator, TransformerMixin):
    """Keep the fitted modeling columns after automatic feature generation."""

    def __init__(self, columns: list[str]):
        self.columns = columns

    def fit(self, X: pd.DataFrame, y: Any = None):
        self.feature_names_in_ = list(pd.DataFrame(X).columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        dataframe = pd.DataFrame(X).copy()
        selected_columns = list(self.columns)
        for column in selected_columns:
            if column not in dataframe.columns:
                dataframe[column] = np.nan
        return dataframe[selected_columns]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(list(self.columns), dtype=object)


def detect_feature_types(dataframe: pd.DataFrame) -> dict[str, list[str]]:
    """Detect feature groups while keeping datetime columns explicit."""

    numerical_columns: list[str] = []
    categorical_columns: list[str] = []
    datetime_columns: list[str] = []

    for column in dataframe.columns:
        series = dataframe[column]
        if is_datetime64_any_dtype(series):
            datetime_columns.append(column)
        elif is_bool_dtype(series):
            categorical_columns.append(column)
        elif is_numeric_dtype(series):
            numerical_columns.append(column)
        elif series.dtype == "object":
            parsed = pd.to_datetime(series, errors="coerce")
            if parsed.notna().mean() >= 0.8 and parsed.nunique(dropna=True) > 1:
                datetime_columns.append(column)
            else:
                categorical_columns.append(column)
        else:
            categorical_columns.append(column)

    return {
        "numerical_columns": numerical_columns,
        "categorical_columns": categorical_columns,
        "datetime_columns": datetime_columns,
    }


def remove_target_column(feature_dict: dict[str, list[str]], target_column: str) -> dict[str, list[str]]:
    return {
        feature_type: [column for column in columns if column != target_column]
        for feature_type, columns in feature_dict.items()
    }


NUMERICAL_IMPUTATION_OPTIONS = ("none", "mean", "median", "most_frequent")
CATEGORICAL_IMPUTATION_OPTIONS = ("none", "most_frequent", "constant")
SCALING_OPTIONS = ("none", "standard", "minmax", "robust")
ENCODING_OPTIONS = ("none", "onehot", "ordinal")


def _get_scaler(scaling_method: str):
    return {
        "standard": StandardScaler(),
        "minmax": MinMaxScaler(),
        "robust": RobustScaler(),
    }.get(scaling_method)


def _get_one_hot_encoder(*, dense_output: bool = False):
    encoder_kwargs = {
        "handle_unknown": "ignore",
    }
    if dense_output:
        try:
            return OneHotEncoder(sparse_output=False, **encoder_kwargs)
        except TypeError:
            return OneHotEncoder(sparse=False, **encoder_kwargs)
    return OneHotEncoder(**encoder_kwargs)


def _finalize_pipeline(steps: list[tuple[str, Any]]) -> Pipeline:
    if not steps:
        steps = [("identity", FunctionTransformer(validate=False))]
    return Pipeline(steps)


def _normalize_column_settings(
    dataframe: pd.DataFrame,
    target_column: str,
    column_settings: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    feature_frame = dataframe.drop(columns=[target_column])
    feature_types = detect_feature_types(feature_frame)
    column_type_lookup: dict[str, str] = {}
    for column_name in feature_types["numerical_columns"]:
        column_type_lookup[column_name] = "numerical"
    for column_name in feature_types["categorical_columns"]:
        column_type_lookup[column_name] = "categorical"
    for column_name in feature_types["datetime_columns"]:
        column_type_lookup[column_name] = "datetime"

    normalized_settings: dict[str, dict[str, Any]] = {}
    for column_name in feature_frame.columns:
        existing_settings = (column_settings or {}).get(column_name, {})
        column_type = existing_settings.get("column_type", column_type_lookup.get(column_name, "categorical"))

        if column_type == "numerical":
            imputation = existing_settings.get("imputation", "median")
            scaling = existing_settings.get("scaling", "standard")
            encoding = existing_settings.get("encoding", "none")
            normalized_settings[column_name] = {
                "column_type": "numerical",
                "drop": bool(existing_settings.get("drop", False)),
                "imputation": imputation if imputation in NUMERICAL_IMPUTATION_OPTIONS else "median",
                "scaling": scaling if scaling in SCALING_OPTIONS else "standard",
                "encoding": encoding if encoding in ENCODING_OPTIONS else "none",
            }
        elif column_type == "datetime":
            imputation = existing_settings.get("imputation", "median")
            scaling = existing_settings.get("scaling", "standard")
            normalized_settings[column_name] = {
                "column_type": "datetime",
                "drop": bool(existing_settings.get("drop", False)),
                "imputation": imputation if imputation in NUMERICAL_IMPUTATION_OPTIONS else "median",
                "scaling": scaling if scaling in SCALING_OPTIONS else "standard",
            }
        else:
            imputation = existing_settings.get("imputation", "most_frequent")
            encoding = existing_settings.get("encoding", "onehot")
            scaling = existing_settings.get("scaling", "none")
            normalized_settings[column_name] = {
                "column_type": "categorical",
                "drop": bool(existing_settings.get("drop", False)),
                "imputation": imputation if imputation in CATEGORICAL_IMPUTATION_OPTIONS else "most_frequent",
                "encoding": encoding if encoding in ENCODING_OPTIONS else "onehot",
                "scaling": scaling if scaling in SCALING_OPTIONS else "none",
            }

    return normalized_settings


def get_default_column_settings(
    dataframe: pd.DataFrame,
    target_column: str,
    column_settings: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Return per-column preprocessing defaults merged with any saved choices."""

    return _normalize_column_settings(dataframe, target_column, column_settings)


def build_numerical_pipeline(
    *,
    scaling_method: str = "standard",
    imputation_strategy: str = "median",
    encoding_method: str = "none",
) -> Pipeline:
    steps: list[tuple[str, Any]] = []
    if imputation_strategy != "none":
        steps.append(("imputer", SimpleImputer(strategy=imputation_strategy)))
    if encoding_method != "none":
        encoder = (
            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            if encoding_method == "ordinal"
            else _get_one_hot_encoder(dense_output=True)
        )
        steps.extend(
            [
                ("caster", CategoricalCaster()),
                ("encoder", encoder),
            ]
        )

    scaler = _get_scaler(scaling_method)
    if scaler is not None:
        steps.append(("scaler", scaler))
    return _finalize_pipeline(steps)


def build_categorical_pipeline(
    *,
    encoding_method: str = "onehot",
    imputation_strategy: str = "most_frequent",
    scaling_method: str = "none",
) -> Pipeline:
    steps: list[tuple[str, Any]] = [("caster", CategoricalCaster())]

    if imputation_strategy != "none":
        steps.append(("imputer", SimpleImputer(strategy=imputation_strategy)))

    if encoding_method != "none":
        encoder = (
            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            if encoding_method == "ordinal"
            else _get_one_hot_encoder(dense_output=scaling_method != "none")
        )
        steps.append(("encoder", encoder))

    scaler = _get_scaler(scaling_method)
    if scaler is not None:
        steps.append(("scaler", scaler))

    return _finalize_pipeline(steps)


def build_datetime_pipeline(
    *,
    scaling_method: str = "standard",
    imputation_strategy: str = "median",
) -> Pipeline:
    steps: list[tuple[str, Any]] = [("extractor", DateTimeFeatureExtractor())]
    if imputation_strategy != "none":
        steps.append(("imputer", SimpleImputer(strategy=imputation_strategy)))
    scaler = _get_scaler(scaling_method)
    if scaler is not None:
        steps.append(("scaler", scaler))
    return _finalize_pipeline(steps)


def build_preprocessor(
    dataframe: pd.DataFrame,
    target_column: str,
    column_settings: dict[str, dict[str, Any]] | None = None,
) -> Pipeline:
    normalized_settings = _normalize_column_settings(dataframe, target_column, column_settings)
    selected_features = [
        column_name
        for column_name, settings in normalized_settings.items()
        if not settings.get("drop", False)
    ]

    if not selected_features:
        raise ValueError("No usable feature columns remain after removing the target column.")

    transformers: list[tuple[str, Pipeline, list[str]]] = []
    for column_name in selected_features:
        settings = normalized_settings[column_name]
        column_type = settings["column_type"]

        if column_type == "numerical":
            transformer = build_numerical_pipeline(
                scaling_method=settings["scaling"],
                imputation_strategy=settings["imputation"],
                encoding_method=settings.get("encoding", "none"),
            )
        elif column_type == "datetime":
            transformer = build_datetime_pipeline(
                scaling_method=settings["scaling"],
                imputation_strategy=settings["imputation"],
            )
        else:
            transformer = build_categorical_pipeline(
                encoding_method=settings["encoding"],
                imputation_strategy=settings["imputation"],
                scaling_method=settings.get("scaling", "none"),
            )

        transformers.append((f"{column_type}_{column_name}", transformer, [column_name]))

    column_transformer = ColumnTransformer(transformers=transformers, remainder="drop")
    pipeline = Pipeline(
        [
            ("column_selector", ColumnSubsetSelector(selected_features)),
            ("column_transformer", column_transformer),
        ]
    )
    pipeline.column_settings_ = normalized_settings
    pipeline.internally_dropped_columns_ = [
        column_name
        for column_name, settings in normalized_settings.items()
        if settings.get("drop", False)
    ]
    pipeline.selected_modeling_columns_ = selected_features
    pipeline.prediction_input_columns_ = selected_features
    return pipeline


def analyze_column_quality(dataframe: pd.DataFrame, target_column: str | None = None) -> list[dict[str, Any]]:
    """Score columns for AutoML usefulness and safe internal dropping."""

    assessments: list[dict[str, Any]] = []
    row_count = max(len(dataframe), 1)
    target_name = (target_column or "").lower()

    for column in dataframe.columns:
        if column == target_column:
            continue

        series = dataframe[column]
        lowered = column.lower()
        unique_count = int(series.nunique(dropna=True))
        unique_ratio = unique_count / row_count
        missing_ratio = float(series.isna().mean())
        reasons: list[str] = []
        auto_drop = False
        severity = "review"

        if missing_ratio >= 0.6:
            reasons.append(f"{missing_ratio:.0%} missing values")
            auto_drop = True
            severity = "drop"

        if lowered in {"id", "index"} or lowered.endswith("id") or lowered.endswith("_id"):
            if unique_ratio >= 0.85 or is_numeric_dtype(series):
                reasons.append("ID-like identifier with little reusable signal")
                auto_drop = True
                severity = "drop"

        if is_numeric_dtype(series) and unique_ratio >= 0.98 and series.is_monotonic_increasing:
            reasons.append("monotonic row identifier")
            auto_drop = True
            severity = "drop"

        if not is_numeric_dtype(series):
            text = series.dropna().astype(str)
            average_length = float(text.str.len().mean()) if not text.empty else 0.0
            if unique_ratio >= 0.5 and unique_count > 50:
                reasons.append("high-cardinality categorical values")
                auto_drop = True
                severity = "drop"
            if average_length >= 18 and unique_ratio >= 0.35:
                reasons.append("free-text-like values that need specialized NLP handling")
                auto_drop = True
                severity = "drop"

        if target_name and lowered != target_name and target_name in lowered:
            reasons.append("name overlaps with the target and may leak outcome information")
            auto_drop = True
            severity = "drop"

        if target_column is not None and target_column in dataframe.columns:
            target_series = dataframe[target_column]
            comparable = pd.concat([series, target_series], axis=1).dropna()
            if not comparable.empty:
                feature_values = comparable.iloc[:, 0].astype(str)
                target_values = comparable.iloc[:, 1].astype(str)
                match_ratio = float((feature_values == target_values).mean())
                if match_ratio >= 0.98:
                    reasons.append("values nearly duplicate the target")
                    auto_drop = True
                    severity = "drop"

        if reasons:
            assessments.append(
                {
                    "column": column,
                    "severity": severity,
                    "auto_drop": auto_drop,
                    "unique_values": unique_count,
                    "missing_ratio": missing_ratio,
                    "reason": "; ".join(dict.fromkeys(reasons)),
                }
            )

    return assessments


def split_features_and_target(
    dataframe: pd.DataFrame,
    target_column: str,
) -> tuple[pd.DataFrame, pd.Series]:
    if target_column not in dataframe.columns:
        raise ValueError(f"Target column '{target_column}' was not found.")
    if dataframe[target_column].isna().any():
        raise ValueError("Target column contains missing values; clean or impute the target before training.")

    X = dataframe.drop(columns=[target_column])
    y = dataframe[target_column]
    if X.empty:
        raise ValueError("At least one feature column is required.")
    if y.nunique(dropna=True) < 2:
        raise ValueError("Target column must contain at least two distinct values.")
    return X, y


def perform_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    problem_type = detect_problem_type(y)
    stratify = y if problem_type == "classification" and y.value_counts().min() >= 2 else None
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )


def _sanitize_feature_names(feature_names: list[str]) -> list[str]:
    cleaned_names: list[str] = []
    name_counts: dict[str, int] = {}
    for raw_name in feature_names:
        normalized_name = (
            str(raw_name)
            .replace("numerical_pipeline__", "")
            .replace("categorical_pipeline__", "")
            .replace("datetime_pipeline__", "")
            .strip()
        )
        duplicate_count = name_counts.get(normalized_name, 0)
        name_counts[normalized_name] = duplicate_count + 1
        cleaned_names.append(
            normalized_name if duplicate_count == 0 else f"{normalized_name}_{duplicate_count + 1}"
        )
    return cleaned_names


def _to_feature_dataframe(transformed_data, preprocessor, index: pd.Index) -> pd.DataFrame:
    if hasattr(transformed_data, "toarray"):
        transformed_data = transformed_data.toarray()
    feature_names = _sanitize_feature_names(list(preprocessor.get_feature_names_out()))
    return pd.DataFrame(transformed_data, columns=feature_names, index=index)


def apply_dataset_cleaning(
    dataframe: pd.DataFrame,
    target_column: str,
    *,
    columns_to_drop: list[str] | None = None,
    drop_rows_with_missing: bool = False,
    row_non_null_threshold: float = 1.0,
    drop_columns_with_missing: bool = False,
    duplicate_handling: str = "keep",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply row and column cleaning before feature-level preprocessing."""

    if target_column not in dataframe.columns:
        raise ValueError(f"Target column '{target_column}' was not found.")

    cleaned_dataframe = dataframe.copy()
    columns_to_drop = [column for column in (columns_to_drop or []) if column in cleaned_dataframe.columns]
    if target_column in columns_to_drop:
        raise ValueError("The target column cannot be dropped during preprocessing.")

    manually_dropped_columns = list(dict.fromkeys(columns_to_drop))
    if manually_dropped_columns:
        cleaned_dataframe = cleaned_dataframe.drop(columns=manually_dropped_columns)

    automatically_dropped_columns: list[str] = []
    if drop_columns_with_missing:
        automatically_dropped_columns = [
            column
            for column in cleaned_dataframe.columns
            if column != target_column and cleaned_dataframe[column].isna().any()
        ]
        if automatically_dropped_columns:
            cleaned_dataframe = cleaned_dataframe.drop(columns=automatically_dropped_columns)

    rows_before = int(dataframe.shape[0])
    columns_before = int(dataframe.shape[1])
    required_non_null_values = None
    duplicate_rows_before = int(cleaned_dataframe.duplicated().sum())

    if drop_rows_with_missing:
        threshold_ratio = min(max(row_non_null_threshold, 0.0), 1.0)
        required_non_null_values = max(
            1,
            math.ceil(len(cleaned_dataframe.columns) * threshold_ratio),
        )
        cleaned_dataframe = cleaned_dataframe.dropna(subset=[target_column])
        cleaned_dataframe = cleaned_dataframe.dropna(
            thresh=required_non_null_values,
            subset=cleaned_dataframe.columns.tolist(),
        )

    duplicate_rows_removed = 0
    if duplicate_handling == "remove":
        rows_before_duplicate_removal = int(cleaned_dataframe.shape[0])
        cleaned_dataframe = cleaned_dataframe.drop_duplicates()
        duplicate_rows_removed = rows_before_duplicate_removal - int(cleaned_dataframe.shape[0])

    cleaning_summary = {
        "original_rows": rows_before,
        "remaining_rows": int(cleaned_dataframe.shape[0]),
        "dropped_rows": rows_before - int(cleaned_dataframe.shape[0]),
        "original_columns": columns_before,
        "remaining_columns": int(cleaned_dataframe.shape[1]),
        "dropped_columns": columns_before - int(cleaned_dataframe.shape[1]),
        "manually_dropped_columns": manually_dropped_columns,
        "columns_dropped_for_missing_values": automatically_dropped_columns,
        "drop_rows_with_missing": drop_rows_with_missing,
        "row_non_null_threshold": row_non_null_threshold if drop_rows_with_missing else None,
        "required_non_null_values": required_non_null_values,
        "drop_columns_with_missing": drop_columns_with_missing,
        "duplicate_rows_before": duplicate_rows_before,
        "duplicate_rows_removed": duplicate_rows_removed,
        "duplicate_handling": duplicate_handling,
        "remaining_missing_values": int(cleaned_dataframe.isna().sum().sum()),
    }

    if cleaned_dataframe.empty:
        raise ValueError("No rows remain after preprocessing. Adjust the row-drop settings and try again.")

    if cleaned_dataframe.shape[1] < 2:
        raise ValueError("No feature columns remain after preprocessing. Adjust the column-drop settings and try again.")

    return cleaned_dataframe, cleaning_summary


def create_model_ready_results(
    dataframe: pd.DataFrame,
    target_column: str,
    *,
    test_size: float = 0.2,
    X_train: pd.DataFrame | None = None,
    X_test: pd.DataFrame | None = None,
    y_train: pd.Series | None = None,
    y_test: pd.Series | None = None,
    X_train_raw: pd.DataFrame | None = None,
    X_test_raw: pd.DataFrame | None = None,
    cleaned_dataframe: pd.DataFrame | None = None,
    fitted_preprocessing_transformer=None,
    cleaning_summary: dict[str, Any] | None = None,
    column_assessments: list[dict[str, Any]] | None = None,
    prediction_input_columns: list[str] | None = None,
) -> dict[str, Any]:
    """Package a dataframe that is already model-ready for downstream pages."""

    transformed_dataframe = dataframe.copy()
    X_full, y_full = split_features_and_target(transformed_dataframe, target_column)

    if any(value is None for value in (X_train, X_test, y_train, y_test)):
        X_train, X_test, y_train, y_test = perform_train_test_split(
            X_full,
            y_full,
            test_size=test_size,
        )

    model_ready_preprocessor = FeatureSchemaAligner().fit(X_train)
    raw_feature_schema = list(X_train_raw.columns) if X_train_raw is not None else list(X_train.columns)

    return {
        "X_train": X_train.copy(),
        "X_test": X_test.copy(),
        "y_train": y_train.copy(),
        "y_test": y_test.copy(),
        "X_train_processed": X_train.copy(),
        "X_test_processed": X_test.copy(),
        "X_train_raw": X_train_raw.copy() if X_train_raw is not None else X_train.copy(),
        "X_test_raw": X_test_raw.copy() if X_test_raw is not None else X_test.copy(),
        "preprocessor": fitted_preprocessing_transformer or model_ready_preprocessor,
        "processed_preprocessor": model_ready_preprocessor,
        "fitted_preprocessing_transformer": fitted_preprocessing_transformer or model_ready_preprocessor,
        "feature_schema": list(X_train.columns),
        "raw_feature_schema": raw_feature_schema,
        "prediction_input_columns": prediction_input_columns or raw_feature_schema,
        "column_assessments": column_assessments or [],
        "internally_dropped_columns": getattr(
            fitted_preprocessing_transformer,
            "internally_dropped_columns_",
            [],
        )
        if fitted_preprocessing_transformer is not None
        else [],
        "selected_modeling_columns": getattr(
            fitted_preprocessing_transformer,
            "selected_modeling_columns_",
            list(X_train.columns),
        )
        if fitted_preprocessing_transformer is not None
        else list(X_train.columns),
        "feature_types": remove_target_column(detect_feature_types(transformed_dataframe), target_column),
        "cleaned_dataframe": cleaned_dataframe.copy() if cleaned_dataframe is not None else transformed_dataframe.copy(),
        "transformed_dataframe": transformed_dataframe.copy(),
        "cleaning_summary": cleaning_summary or {
            "original_rows": int(transformed_dataframe.shape[0]),
            "remaining_rows": int(transformed_dataframe.shape[0]),
            "dropped_rows": 0,
            "original_columns": int(transformed_dataframe.shape[1]),
            "remaining_columns": int(transformed_dataframe.shape[1]),
            "dropped_columns": 0,
            "manually_dropped_columns": [],
            "columns_dropped_for_missing_values": [],
            "drop_rows_with_missing": False,
            "row_non_null_threshold": None,
            "required_non_null_values": None,
            "drop_columns_with_missing": False,
            "remaining_missing_values": int(transformed_dataframe.isna().sum().sum()),
        },
    }


def execute_preprocessing_pipeline(
    dataframe: pd.DataFrame,
    target_column: str,
    column_settings: dict[str, dict[str, Any]] | None = None,
    test_size: float = 0.2,
    columns_to_drop: list[str] | None = None,
    drop_rows_with_missing: bool = False,
    row_non_null_threshold: float = 1.0,
    drop_columns_with_missing: bool = False,
    duplicate_handling: str = "keep",
) -> dict[str, Any]:
    cleaned_dataframe, cleaning_summary = apply_dataset_cleaning(
        dataframe,
        target_column,
        columns_to_drop=columns_to_drop,
        drop_rows_with_missing=drop_rows_with_missing,
        row_non_null_threshold=row_non_null_threshold,
        drop_columns_with_missing=drop_columns_with_missing,
        duplicate_handling=duplicate_handling,
    )

    X_raw, y = split_features_and_target(cleaned_dataframe, target_column)
    X_train_raw, X_test_raw, y_train, y_test = perform_train_test_split(X_raw, y, test_size=test_size)

    fitted_preprocessor = build_preprocessor(
        dataframe=cleaned_dataframe,
        target_column=target_column,
        column_settings=column_settings,
    )

    X_train_processed = fitted_preprocessor.fit_transform(X_train_raw)
    X_test_processed = fitted_preprocessor.transform(X_test_raw)
    X_full_processed = fitted_preprocessor.transform(X_raw)

    X_train_dataframe = _to_feature_dataframe(
        X_train_processed,
        fitted_preprocessor,
        X_train_raw.index,
    )
    X_test_dataframe = _to_feature_dataframe(
        X_test_processed,
        fitted_preprocessor,
        X_test_raw.index,
    )
    X_full_dataframe = _to_feature_dataframe(
        X_full_processed,
        fitted_preprocessor,
        X_raw.index,
    )

    transformed_dataframe = pd.concat(
        [
            X_full_dataframe,
            y.loc[X_full_dataframe.index].rename(target_column),
        ],
        axis=1,
    )

    cleaning_summary["internally_dropped_columns"] = getattr(
        fitted_preprocessor,
        "internally_dropped_columns_",
        [],
    )
    cleaning_summary["column_settings"] = getattr(fitted_preprocessor, "column_settings_", {})

    return create_model_ready_results(
        transformed_dataframe,
        target_column,
        test_size=test_size,
        X_train=X_train_dataframe,
        X_test=X_test_dataframe,
        y_train=y_train,
        y_test=y_test,
        X_train_raw=X_train_raw,
        X_test_raw=X_test_raw,
        cleaned_dataframe=cleaned_dataframe,
        fitted_preprocessing_transformer=fitted_preprocessor,
        cleaning_summary=cleaning_summary,
        column_assessments=analyze_column_quality(cleaned_dataframe, target_column=target_column),
        prediction_input_columns=getattr(fitted_preprocessor, "prediction_input_columns_", list(X_raw.columns)),
    )


def generate_preprocessing_recommendations(
    dataframe: pd.DataFrame,
    target_column: str | None = None,
) -> list[str]:
    recommendations: list[str] = []
    missing_ratio = dataframe.isna().sum().sum() / max(dataframe.size, 1)
    if missing_ratio > 0.2:
        recommendations.append(
            "High missing-value ratio detected; review per-column imputers carefully before training."
        )
    if dataframe.duplicated().any():
        duplicate_recommendation = assess_duplicate_row_recommendation(dataframe)
        recommendations.append(
            f"{duplicate_recommendation['headline']} {duplicate_recommendation['reasoning']}"
        )

    for assessment in analyze_column_quality(dataframe, target_column=target_column):
        action = "consider dropping or reworking" if assessment["auto_drop"] else "review manually"
        recommendations.append(f"Column '{assessment['column']}' may need attention: {action} because {assessment['reason']}.")

    for column in dataframe.select_dtypes(include="number").columns:
        if column == target_column:
            continue
        skewness = dataframe[column].dropna().skew()
        if pd.notna(skewness) and abs(skewness) > 1:
            recommendations.append(
                f"Column '{column}' is highly skewed; robust scaling or a later feature transform may help."
            )
    return recommendations


def generate_preprocessing_summary(
    column_settings: dict[str, dict[str, Any]],
    *,
    drop_rows_with_missing: bool = False,
    row_non_null_threshold: float | None = None,
    drop_columns_with_missing: bool = False,
    columns_to_drop: list[str] | None = None,
    duplicate_handling: str = "keep",
) -> dict[str, str]:
    scaling_summary: dict[str, int] = {}
    encoding_summary: dict[str, int] = {}
    imputation_summary: dict[str, int] = {}

    for settings in column_settings.values():
        if settings.get("drop", False):
            continue

        imputation_key = settings.get("imputation", "not_set")
        imputation_summary[imputation_key] = imputation_summary.get(imputation_key, 0) + 1

        if settings.get("column_type") in {"numerical", "datetime"}:
            scaling_key = settings.get("scaling", "none")
            scaling_summary[scaling_key] = scaling_summary.get(scaling_key, 0) + 1
        if settings.get("column_type") == "categorical":
            encoding_key = settings.get("encoding", "onehot")
            encoding_summary[encoding_key] = encoding_summary.get(encoding_key, 0) + 1
            scaling_key = settings.get("scaling", "none")
            scaling_summary[scaling_key] = scaling_summary.get(scaling_key, 0) + 1
        if settings.get("column_type") == "numerical":
            encoding_key = settings.get("encoding", "none")
            encoding_summary[encoding_key] = encoding_summary.get(encoding_key, 0) + 1

    return {
        "Active Feature Columns": str(sum(not settings.get("drop", False) for settings in column_settings.values())),
        "Scaling Choices": ", ".join(f"{name}: {count}" for name, count in scaling_summary.items()) or "None",
        "Encoding Choices": ", ".join(f"{name}: {count}" for name, count in encoding_summary.items()) or "None",
        "Imputation Choices": ", ".join(f"{name}: {count}" for name, count in imputation_summary.items()) or "None",
        "Drop Rows With Missing Values": "Yes" if drop_rows_with_missing else "No",
        "Row Retention Threshold": (
            f"{int((row_non_null_threshold or 0) * 100)}% non-null values required"
            if drop_rows_with_missing and row_non_null_threshold is not None
            else "Not Applied"
        ),
        "Drop Columns With Missing Values": "Yes" if drop_columns_with_missing else "No",
        "Duplicate Handling": "Remove duplicates" if duplicate_handling == "remove" else "Keep duplicates",
        "Selected Columns To Drop": ", ".join(columns_to_drop or []) or "None",
    }
