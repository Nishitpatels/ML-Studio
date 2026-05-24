"""Explainability helpers with safe fallbacks for unsupported models."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px

try:
    import shap
except Exception:  # pragma: no cover - optional runtime dependency
    shap = None


def extract_model_from_pipeline(pipeline):
    return pipeline.named_steps["model"]


def extract_preprocessor_from_pipeline(pipeline):
    return pipeline.named_steps["preprocessor"]


def transform_features(pipeline, X: pd.DataFrame):
    preprocessor = extract_preprocessor_from_pipeline(pipeline)
    schema = pipeline.named_steps.get("schema")
    transformed_input = schema.transform(X) if schema is not None else X
    return preprocessor.transform(transformed_input)


def get_transformed_feature_names(pipeline, original_dataframe: pd.DataFrame) -> list[str]:
    preprocessor = extract_preprocessor_from_pipeline(pipeline)
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        transformed = transform_features(pipeline, original_dataframe.head(1))
        return [f"feature_{index}" for index in range(transformed.shape[1])]


def get_feature_importance(pipeline, original_dataframe: pd.DataFrame) -> pd.DataFrame | None:
    model = extract_model_from_pipeline(pipeline)
    feature_names = get_transformed_feature_names(pipeline, original_dataframe)

    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_)
    elif hasattr(model, "coef_"):
        coefficients = np.asarray(model.coef_)
        importances = np.abs(coefficients[0] if coefficients.ndim > 1 else coefficients)
    else:
        return None

    if len(feature_names) != len(importances):
        feature_names = [f"feature_{index}" for index in range(len(importances))]

    return (
        pd.DataFrame({"Feature": feature_names, "Importance": importances})
        .sort_values(by="Importance", ascending=False)
        .reset_index(drop=True)
    )


def create_feature_importance_chart(
    importance_dataframe: pd.DataFrame,
    top_n: int = 20,
):
    top_features = importance_dataframe.head(top_n).sort_values("Importance")
    figure = px.bar(
        top_features,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Top Feature Importances",
    )
    figure.update_layout(height=700)
    return figure


def create_shap_explainer(pipeline, X_train: pd.DataFrame):
    if shap is None:
        return None

    model = extract_model_from_pipeline(pipeline)
    transformed_X = transform_features(pipeline, X_train)
    sample = transformed_X[: min(200, transformed_X.shape[0])]

    try:
        if hasattr(model, "feature_importances_"):
            return shap.TreeExplainer(model)
        if hasattr(model, "coef_"):
            return shap.LinearExplainer(model, sample)
    except Exception:
        return None
    return None


def generate_shap_values(pipeline, X_train: pd.DataFrame):
    explainer = create_shap_explainer(pipeline, X_train)
    if explainer is None:
        return None

    transformed_X = transform_features(pipeline, X_train)
    sample = transformed_X[: min(200, transformed_X.shape[0])]
    try:
        values = explainer.shap_values(sample)
        if isinstance(values, list):
            values = values[0]
        if hasattr(values, "values"):
            values = values.values
        values = np.asarray(values)
        if values.ndim == 3:
            values = values[:, :, 0]
        return values
    except Exception:
        return None


def generate_shap_summary_dataframe(shap_values, feature_names: list[str]) -> pd.DataFrame:
    values = np.asarray(shap_values)
    if values.ndim != 2:
        raise ValueError("SHAP values must be a 2D array.")
    if values.shape[1] != len(feature_names):
        feature_names = [f"feature_{index}" for index in range(values.shape[1])]
    return (
        pd.DataFrame(
            {
                "Feature": feature_names,
                "Mean Absolute SHAP": np.abs(values).mean(axis=0),
            }
        )
        .sort_values(by="Mean Absolute SHAP", ascending=False)
        .reset_index(drop=True)
    )


def create_shap_importance_chart(shap_dataframe: pd.DataFrame, top_n: int = 20):
    top_features = shap_dataframe.head(top_n).sort_values("Mean Absolute SHAP")
    figure = px.bar(
        top_features,
        x="Mean Absolute SHAP",
        y="Feature",
        orientation="h",
        title="Top SHAP Feature Importances",
    )
    figure.update_layout(height=700)
    return figure


def explain_single_prediction(pipeline, X: pd.DataFrame, row_index: int = 0):
    shap_values = generate_shap_values(pipeline, X)
    if shap_values is None or row_index >= len(shap_values):
        return None
    return pd.Series(shap_values[row_index])


def generate_explainability_summary(pipeline, original_dataframe: pd.DataFrame) -> dict[str, Any]:
    model = extract_model_from_pipeline(pipeline)
    transformed_feature_count = transform_features(pipeline, original_dataframe.head(1)).shape[1]
    return {
        "Model Name": model.__class__.__name__,
        "Supports Feature Importance": get_feature_importance(pipeline, original_dataframe) is not None,
        "Supports SHAP": create_shap_explainer(pipeline, original_dataframe) is not None,
        "Total Features": int(transformed_feature_count),
    }
