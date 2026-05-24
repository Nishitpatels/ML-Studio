"""Dataset analysis utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.data_loader import calculate_dataset_health_score, get_datetime_columns
from src.model_selector import detect_problem_type


def get_basic_dataset_info(dataframe: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": int(dataframe.shape[0]),
        "columns": int(dataframe.shape[1]),
        "memory_usage_mb": round(dataframe.memory_usage(deep=True).sum() / (1024 * 1024), 2),
        "total_missing_values": int(dataframe.isna().sum().sum()),
        "duplicate_rows": int(dataframe.duplicated().sum()),
    }


def detect_column_types(dataframe: pd.DataFrame) -> dict[str, list[str]]:
    datetime_columns = get_datetime_columns(dataframe)
    numerical_columns = dataframe.select_dtypes(include="number").columns.tolist()
    categorical_columns = [
        column
        for column in dataframe.columns
        if column not in numerical_columns and column not in datetime_columns
    ]
    return {
        "numerical_columns": numerical_columns,
        "categorical_columns": categorical_columns,
        "datetime_columns": datetime_columns,
    }


def analyze_missing_values(dataframe: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Column": dataframe.columns,
            "Missing Values": dataframe.isna().sum().values,
            "Missing Percentage": dataframe.isna().mean().values * 100,
        }
    ).sort_values(by="Missing Percentage", ascending=False)


def analyze_duplicates(dataframe: pd.DataFrame) -> dict[str, float | int]:
    duplicate_count = int(dataframe.duplicated().sum())
    duplicate_percentage = round((duplicate_count / max(len(dataframe), 1)) * 100, 2)
    return {
        "duplicate_count": duplicate_count,
        "duplicate_percentage": duplicate_percentage,
    }


def assess_duplicate_row_recommendation(dataframe: pd.DataFrame) -> dict[str, Any]:
    """Recommend whether duplicate rows should be removed, retained, or reviewed."""

    duplicate_summary = analyze_duplicates(dataframe)
    duplicate_count = int(duplicate_summary["duplicate_count"])
    duplicate_percentage = float(duplicate_summary["duplicate_percentage"])
    row_count = int(len(dataframe))

    if duplicate_count == 0:
        return {
            "action": "retain",
            "headline": "No exact duplicate rows were detected.",
            "reasoning": "The dataset does not contain repeated full rows, so no duplicate cleanup is needed.",
        }

    lowered_columns = [column.lower() for column in dataframe.columns]
    transaction_like_keywords = {
        "transaction",
        "invoice",
        "order",
        "purchase",
        "payment",
        "event",
        "session",
        "visit",
        "click",
        "log",
        "timestamp",
        "date",
        "time",
    }
    transaction_like = any(
        any(keyword in column_name for keyword in transaction_like_keywords)
        for column_name in lowered_columns
    ) or len(get_datetime_columns(dataframe)) > 0

    if transaction_like:
        if duplicate_percentage <= 1 and row_count >= 1000:
            return {
                "action": "retain",
                "headline": "Duplicates can likely be retained for now.",
                "reasoning": (
                    f"Only {duplicate_percentage}% of rows are duplicated, and the dataset includes "
                    "transactional or time-based signals. These repeats may represent legitimate recurring events."
                ),
            }
        return {
            "action": "review",
            "headline": "Duplicates should be reviewed before removal.",
            "reasoning": (
                f"{duplicate_count} duplicate rows ({duplicate_percentage}%) were found, but the dataset looks "
                "transactional or event-based. Keep them unless business context confirms they are accidental duplicates."
            ),
        }

    if duplicate_percentage >= 10 or duplicate_count >= 100:
        return {
            "action": "remove",
            "headline": "Duplicates should likely be removed.",
            "reasoning": (
                f"{duplicate_count} duplicate rows ({duplicate_percentage}%) is high for a record-style dataset, "
                "so repeated rows are more likely to be accidental copies than meaningful events."
            ),
        }

    if row_count <= 500:
        return {
            "action": "review",
            "headline": "Duplicates should be reviewed carefully.",
            "reasoning": (
                f"The dataset is relatively small ({row_count} rows), so even {duplicate_count} repeated rows can "
                "materially change the analysis. Review them before deciding whether to remove them."
            ),
        }

    return {
        "action": "remove",
        "headline": "Duplicates are likely safe to remove.",
        "reasoning": (
            f"{duplicate_count} duplicate rows ({duplicate_percentage}%) were detected and the dataset does not look "
            "transactional. Exact repeated rows in this context usually indicate redundant records."
        ),
    }


def analyze_cardinality(dataframe: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Column": dataframe.columns,
            "Unique Values": dataframe.nunique(dropna=True).values,
            "Cardinality Percentage": dataframe.nunique(dropna=True).values / max(len(dataframe), 1) * 100,
        }
    ).sort_values(by="Unique Values", ascending=False)


def analyze_skewness(dataframe: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for column in dataframe.select_dtypes(include="number").columns:
        skewness = dataframe[column].dropna().skew()
        if pd.isna(skewness):
            interpretation = "Insufficient Data"
        elif skewness > 1:
            interpretation = "Highly Positively Skewed"
        elif skewness < -1:
            interpretation = "Highly Negatively Skewed"
        elif abs(skewness) <= 0.5:
            interpretation = "Approximately Normal"
        else:
            interpretation = "Moderately Skewed"
        rows.append(
            {
                "Column": column,
                "Skewness": None if pd.isna(skewness) else round(float(skewness), 4),
                "Interpretation": interpretation,
            }
        )
    return pd.DataFrame(rows)


def detect_outliers_iqr(dataframe: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for column in dataframe.select_dtypes(include="number").columns:
        series = dataframe[column].dropna()
        if series.empty:
            continue
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outlier_count = int(((series < lower_bound) | (series > upper_bound)).sum())
        rows.append(
            {
                "Column": column,
                "Outlier Count": outlier_count,
                "Outlier Percentage": round((outlier_count / max(len(dataframe), 1)) * 100, 2),
            }
        )
    return pd.DataFrame(rows)


def get_correlation_matrix(dataframe: pd.DataFrame) -> pd.DataFrame:
    numerical_dataframe = dataframe.select_dtypes(include="number")
    return numerical_dataframe.corr() if not numerical_dataframe.empty else pd.DataFrame()


def detect_high_correlations(
    dataframe: pd.DataFrame,
    threshold: float = 0.8,
) -> pd.DataFrame:
    correlation_matrix = get_correlation_matrix(dataframe)
    rows: list[dict[str, Any]] = []
    for row_index in range(len(correlation_matrix.columns)):
        for column_index in range(row_index):
            correlation_value = correlation_matrix.iloc[row_index, column_index]
            if pd.notna(correlation_value) and abs(correlation_value) >= threshold:
                rows.append(
                    {
                        "Feature 1": correlation_matrix.columns[row_index],
                        "Feature 2": correlation_matrix.columns[column_index],
                        "Correlation": round(float(correlation_value), 4),
                    }
                )
    return pd.DataFrame(rows)


def analyze_target_column(
    dataframe: pd.DataFrame,
    target_column: str,
) -> dict[str, Any]:
    target_series = dataframe[target_column]
    return {
        "target_column": target_column,
        "problem_type": detect_problem_type(target_series).title(),
        "unique_values": int(target_series.nunique(dropna=True)),
        "missing_values": int(target_series.isna().sum()),
        "data_type": str(target_series.dtype),
    }


def analyze_class_balance(
    dataframe: pd.DataFrame,
    target_column: str,
) -> pd.DataFrame:
    distribution = dataframe[target_column].value_counts(dropna=False)
    return pd.DataFrame(
        {
            "Class": distribution.index.astype(str),
            "Count": distribution.values,
            "Percentage": np.round(distribution.values / max(len(dataframe), 1) * 100, 2),
        }
    )


def generate_dataset_recommendations(dataframe: pd.DataFrame) -> list[str]:
    recommendations: list[str] = []
    missing_ratio = dataframe.isna().sum().sum() / max(dataframe.size, 1)
    if missing_ratio > 0.2:
        recommendations.append("Consider a stronger missing-value strategy before modeling.")
    if dataframe.duplicated().any():
        duplicate_recommendation = assess_duplicate_row_recommendation(dataframe)
        recommendations.append(
            f"{duplicate_recommendation['headline']} {duplicate_recommendation['reasoning']}"
        )
    for column in dataframe.columns:
        if dataframe[column].nunique(dropna=True) / max(len(dataframe), 1) > 0.8:
            recommendations.append(f"Column '{column}' has high cardinality and may need careful encoding.")
    return recommendations


def perform_complete_analysis(
    dataframe: pd.DataFrame,
    target_column: str | None = None,
) -> dict[str, Any]:
    results = {
        "basic_info": get_basic_dataset_info(dataframe),
        "column_types": detect_column_types(dataframe),
        "missing_analysis": analyze_missing_values(dataframe),
        "duplicate_analysis": analyze_duplicates(dataframe),
        "duplicate_recommendation": assess_duplicate_row_recommendation(dataframe),
        "cardinality_analysis": analyze_cardinality(dataframe),
        "skewness_analysis": analyze_skewness(dataframe),
        "outlier_analysis": detect_outliers_iqr(dataframe),
        "correlation_matrix": get_correlation_matrix(dataframe),
        "high_correlations": detect_high_correlations(dataframe),
        "health_score": calculate_dataset_health_score(dataframe),
        "recommendations": generate_dataset_recommendations(dataframe),
    }
    if target_column is not None and target_column in dataframe.columns:
        results["target_analysis"] = analyze_target_column(dataframe, target_column)
        if results["target_analysis"]["problem_type"] == "Classification":
            results["class_balance"] = analyze_class_balance(dataframe, target_column)
    return results
