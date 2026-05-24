"""Training helpers for classification and regression workflows."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, precision_score, r2_score, recall_score
from sklearn.pipeline import Pipeline

from src.model_selector import detect_problem_type, get_models_by_problem_type
from src.preprocessing import FeatureSchemaAligner


def build_training_pipeline(preprocessor, model, feature_columns: list[str] | None = None) -> Pipeline:
    steps: list[tuple[str, Any]] = []
    if feature_columns is not None:
        aligner = FeatureSchemaAligner()
        aligner.feature_names_in_ = list(feature_columns)
        steps.append(("schema", aligner))
    steps.extend(
        [
            ("preprocessor", clone(preprocessor)),
            ("model", clone(model)),
        ]
    )
    return Pipeline(steps)


def evaluate_classification_model(y_true, y_pred) -> dict[str, float]:
    return {
        "Accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "Precision": round(float(precision_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "Recall": round(float(recall_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "F1 Score": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
    }


def evaluate_regression_model(y_true, y_pred) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "MSE": round(float(mse), 4),
        "RMSE": round(float(np.sqrt(mse)), 4),
        "R2 Score": round(float(r2_score(y_true, y_pred)), 4),
    }


def train_single_model(
    model_name: str,
    model,
    preprocessor,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    problem_type: str,
) -> dict[str, Any]:
    pipeline = build_training_pipeline(preprocessor, model, list(X_train.columns))
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    metrics = (
        evaluate_classification_model(y_test, predictions)
        if problem_type == "classification"
        else evaluate_regression_model(y_test, predictions)
    )
    primary_metric = metrics["Accuracy"] if problem_type == "classification" else metrics["R2 Score"]
    return {
        "model_name": model_name,
        "pipeline": pipeline,
        "metrics": metrics,
        "predictions": predictions,
        "primary_metric": primary_metric,
    }


def train_all_models(
    dataframe: pd.DataFrame,
    target_column: str,
    preprocessing_results: dict[str, Any],
    selected_models: list[str] | None = None,
) -> dict[str, Any]:
    problem_type = detect_problem_type(dataframe[target_column])
    models = get_models_by_problem_type(problem_type)
    if selected_models is not None:
        models = {name: model for name, model in models.items() if name in selected_models}
    if not models:
        raise ValueError("No valid models were selected for training.")

    X_train = preprocessing_results.get("X_train_raw", preprocessing_results["X_train"])
    X_test = preprocessing_results.get("X_test_raw", preprocessing_results["X_test"])
    y_train = preprocessing_results["y_train"]
    y_test = preprocessing_results["y_test"]
    preprocessor = preprocessing_results.get(
        "fitted_preprocessing_transformer",
        preprocessing_results["preprocessor"],
    )

    trained_models: dict[str, dict[str, Any]] = {}
    evaluation_rows: list[dict[str, Any]] = []
    errors: dict[str, str] = {}

    for model_name, model in models.items():
        try:
            results = train_single_model(
                model_name=model_name,
                model=model,
                preprocessor=preprocessor,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                problem_type=problem_type,
            )
            trained_models[model_name] = results
            evaluation_rows.append({"Model": model_name, **results["metrics"], "Primary Metric": results["primary_metric"]})
        except Exception as error:
            errors[model_name] = str(error)
            evaluation_rows.append({"Model": model_name, "Error": str(error)})

    evaluation_dataframe = pd.DataFrame(evaluation_rows)
    best_model_name = select_best_model(evaluation_dataframe, problem_type)
    return {
        "problem_type": problem_type,
        "trained_models": trained_models,
        "evaluation_dataframe": evaluation_dataframe,
        "best_model_name": best_model_name,
        "best_model": trained_models.get(best_model_name),
        "errors": errors,
    }


def select_best_model(evaluation_dataframe: pd.DataFrame, problem_type: str) -> str | None:
    metric = "Accuracy" if problem_type == "classification" else "R2 Score"
    if metric not in evaluation_dataframe.columns:
        return None
    valid_results = evaluation_dataframe.dropna(subset=[metric])
    if valid_results.empty:
        return None
    return str(valid_results.loc[valid_results[metric].idxmax(), "Model"])


def train_best_model_only(
    dataframe: pd.DataFrame,
    target_column: str,
    preprocessing_results: dict[str, Any],
) -> dict[str, Any]:
    problem_type = detect_problem_type(dataframe[target_column])
    best_model_name = "Random Forest" if problem_type == "classification" else "Random Forest Regressor"
    return train_single_model(
        model_name=best_model_name,
        model=get_models_by_problem_type(problem_type)[best_model_name],
        preprocessor=preprocessing_results.get(
            "fitted_preprocessing_transformer",
            preprocessing_results["preprocessor"],
        ),
        X_train=preprocessing_results.get("X_train_raw", preprocessing_results["X_train"]),
        X_test=preprocessing_results.get("X_test_raw", preprocessing_results["X_test"]),
        y_train=preprocessing_results["y_train"],
        y_test=preprocessing_results["y_test"],
        problem_type=problem_type,
    )


def generate_training_summary(training_results: dict[str, Any]) -> dict[str, Any]:
    return {
        "Problem Type": training_results["problem_type"],
        "Total Models Trained": len(training_results["trained_models"]),
        "Best Model": training_results["best_model_name"],
    }
