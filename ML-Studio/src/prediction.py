"""Prediction helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def get_expected_features(pipeline) -> list[str]:
    if "schema" in pipeline.named_steps:
        return list(pipeline.named_steps["schema"].feature_names_in_)
    preprocessor = pipeline.named_steps.get("preprocessor")
    if preprocessor is not None and hasattr(preprocessor, "feature_names_in_"):
        return list(preprocessor.feature_names_in_)
    return []


def prepare_prediction_input(
    input_dataframe: pd.DataFrame,
    expected_features: list[str],
) -> tuple[pd.DataFrame, list[str], list[str]]:
    prepared = input_dataframe.copy()
    missing_columns = [column for column in expected_features if column not in prepared.columns]
    extra_columns = [column for column in prepared.columns if column not in expected_features]
    for column in missing_columns:
        prepared[column] = np.nan
    if expected_features:
        prepared = prepared[expected_features]
    return prepared, missing_columns, extra_columns


def make_single_prediction(pipeline, input_dataframe: pd.DataFrame):
    prediction = pipeline.predict(input_dataframe)
    return prediction[0]


def get_prediction_probabilities(pipeline, input_dataframe: pd.DataFrame):
    return pipeline.predict_proba(input_dataframe)[0] if hasattr(pipeline, "predict_proba") else None


def make_batch_predictions(pipeline, input_dataframe: pd.DataFrame) -> pd.DataFrame:
    prediction_dataframe = input_dataframe.copy()
    prediction_dataframe["Prediction"] = pipeline.predict(input_dataframe)
    return prediction_dataframe


def make_batch_predictions_with_probabilities(pipeline, input_dataframe: pd.DataFrame) -> pd.DataFrame:
    prediction_dataframe = make_batch_predictions(pipeline, input_dataframe)
    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(input_dataframe)
        prediction_dataframe["Confidence"] = np.max(probabilities, axis=1)
    return prediction_dataframe


def validate_prediction_input(input_dataframe: pd.DataFrame, training_columns) -> list[str]:
    expected_features = list(training_columns)
    return [column for column in expected_features if column not in input_dataframe.columns]


def format_prediction_output(prediction) -> str:
    return str(prediction)


def _format_target_name(target_column: str | None) -> str:
    if not target_column:
        return "Outcome"
    return target_column.replace("_", " ").strip().title()


def interpret_prediction_label(prediction, target_column: str | None = None) -> str:
    """Convert a raw prediction into a more human-readable label."""

    target_name = _format_target_name(target_column)
    lowered_target = (target_column or "").strip().lower()
    positive_values = {1, 1.0, True, "1", "1.0", "true", "yes", "y"}
    negative_values = {0, 0.0, False, "0", "0.0", "false", "no", "n"}
    normalized_prediction = str(prediction).strip().lower()

    if prediction in positive_values or normalized_prediction in positive_values:
        if lowered_target == "survived":
            return "Survived (Yes)"
        return f"{target_name} (Yes)"

    if prediction in negative_values or normalized_prediction in negative_values:
        if lowered_target == "survived":
            return "Did Not Survive (No)"
        return f"{target_name} (No)"

    return str(prediction)


def build_prediction_explanation(
    prediction,
    *,
    target_column: str | None = None,
    probabilities=None,
    classes=None,
) -> dict[str, str | float | None]:
    """Package a readable prediction summary for the UI."""

    interpretation = interpret_prediction_label(prediction, target_column)
    confidence = None
    if probabilities is not None:
        confidence = float(np.max(probabilities))

    probability_note = (
        f" The model is most confident in this outcome with probability {confidence:.2%}."
        if confidence is not None
        else ""
    )

    if classes is not None and probabilities is not None:
        classes_list = [str(item) for item in classes]
        explanation = (
            f"The predicted class is '{prediction}', interpreted here as '{interpretation}'."
            f"{probability_note} Available classes: {', '.join(classes_list)}."
        )
    else:
        explanation = (
            f"The predicted output is '{prediction}', interpreted here as '{interpretation}'."
            f"{probability_note}"
        )

    return {
        "raw_prediction": str(prediction),
        "interpretation": interpretation,
        "confidence": confidence,
        "explanation": explanation.strip(),
    }


def generate_prediction_summary(prediction_dataframe: pd.DataFrame) -> dict[str, int]:
    return {
        "Total Predictions": len(prediction_dataframe),
        "Prediction Columns": prediction_dataframe.shape[1],
    }
