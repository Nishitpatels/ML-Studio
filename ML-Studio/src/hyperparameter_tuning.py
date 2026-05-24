"""Hyperparameter tuning helpers."""

from __future__ import annotations

import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from src.model_selector import detect_problem_type
from src.training import build_training_pipeline, evaluate_classification_model, evaluate_regression_model


def get_parameter_grids() -> dict[str, dict[str, list]]:
    return {
        "Random Forest": {"model__n_estimators": [50, 100], "model__max_depth": [None, 5, 10]},
        "Gradient Boosting": {"model__n_estimators": [50, 100], "model__learning_rate": [0.01, 0.1]},
        "Decision Tree": {"model__max_depth": [None, 5, 10], "model__min_samples_split": [2, 5]},
        "Logistic Regression": {"model__C": [0.1, 1.0, 10.0]},
        "Random Forest Regressor": {"model__n_estimators": [50, 100], "model__max_depth": [None, 5, 10]},
        "Gradient Boosting Regressor": {"model__n_estimators": [50, 100], "model__learning_rate": [0.01, 0.1]},
        "Decision Tree Regressor": {"model__max_depth": [None, 5, 10], "model__min_samples_split": [2, 5]},
        "Ridge Regression": {"model__alpha": [0.1, 1.0, 10.0]},
        "Lasso Regression": {"model__alpha": [0.001, 0.01, 0.1]},
    }


def perform_grid_search(pipeline, parameter_grid, X_train, y_train, scoring="accuracy", cv=3):
    search = GridSearchCV(pipeline, parameter_grid, scoring=scoring, cv=cv, n_jobs=-1)
    search.fit(X_train, y_train)
    return search


def perform_random_search(pipeline, parameter_grid, X_train, y_train, scoring="accuracy", cv=3, n_iter=5):
    search = RandomizedSearchCV(
        pipeline,
        parameter_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        n_iter=min(n_iter, max(1, sum(len(values) for values in parameter_grid.values()))),
        random_state=42,
    )
    search.fit(X_train, y_train)
    return search


def tune_model(
    model_name,
    model,
    preprocessor,
    X_train,
    y_train,
    tuning_method="grid",
):
    parameter_grid = get_parameter_grids().get(model_name)
    if parameter_grid is None:
        return None

    pipeline = build_training_pipeline(preprocessor, clone(model), list(X_train.columns))
    scoring = "accuracy" if detect_problem_type(y_train) == "classification" else "r2"
    tuner = (
        perform_grid_search(pipeline, parameter_grid, X_train, y_train, scoring=scoring)
        if tuning_method == "grid"
        else perform_random_search(pipeline, parameter_grid, X_train, y_train, scoring=scoring)
    )
    cv_results = pd.DataFrame(tuner.cv_results_).copy()
    cv_results["candidate_id"] = range(1, len(cv_results) + 1)
    return {
        "model_name": model_name,
        "best_estimator": tuner.best_estimator_,
        "best_score": tuner.best_score_,
        "best_parameters": tuner.best_params_,
        "cv_results": cv_results,
        "scoring_metric": scoring,
        "tuning_method": tuning_method,
    }


def build_tuned_model_results(
    model_name,
    tuning_results,
    X_test,
    y_test,
    problem_type,
) -> dict:
    """Convert a fitted tuner result into the active model-result contract."""

    pipeline = tuning_results["best_estimator"]
    predictions = pipeline.predict(X_test)
    metrics = (
        evaluate_classification_model(y_test, predictions)
        if problem_type == "classification"
        else evaluate_regression_model(y_test, predictions)
    )
    primary_metric = metrics["Accuracy"] if problem_type == "classification" else metrics["R2 Score"]
    return {
        "model_name": f"{model_name} (Tuned)",
        "pipeline": pipeline,
        "metrics": metrics,
        "predictions": predictions,
        "primary_metric": primary_metric,
    }


def merge_tuned_model_with_training_results(
    existing_training_results,
    tuned_model_results,
    problem_type,
) -> dict:
    """Promote the tuned model while preserving any existing leaderboard rows."""

    model_name = tuned_model_results["model_name"]
    trained_models = {}
    evaluation_dataframe = pd.DataFrame()
    errors = {}

    if existing_training_results:
        trained_models.update(existing_training_results.get("trained_models", {}))
        evaluation_dataframe = existing_training_results.get("evaluation_dataframe", pd.DataFrame())
        errors.update(existing_training_results.get("errors", {}))

    trained_models[model_name] = tuned_model_results
    tuned_row = pd.DataFrame(
        [
            {
                "Model": model_name,
                **tuned_model_results["metrics"],
                "Primary Metric": tuned_model_results["primary_metric"],
            }
        ]
    )
    if not evaluation_dataframe.empty and "Model" in evaluation_dataframe.columns:
        evaluation_dataframe = evaluation_dataframe[evaluation_dataframe["Model"] != model_name]
    evaluation_dataframe = pd.concat([evaluation_dataframe, tuned_row], ignore_index=True)

    return {
        "problem_type": problem_type,
        "trained_models": trained_models,
        "evaluation_dataframe": evaluation_dataframe,
        "best_model_name": model_name,
        "best_model": tuned_model_results,
        "errors": errors,
    }


def get_parameter_columns(cv_results: pd.DataFrame) -> list[str]:
    """Return parameter columns from sklearn CV results."""

    return [column for column in cv_results.columns if column.startswith("param_")]


def generate_tuning_explanation(tuning_results: dict) -> list[str]:
    """Generate a simple explanation for the best tuning outcome."""

    cv_results = tuning_results["cv_results"]
    best_score = float(tuning_results["best_score"])
    score_column = "mean_test_score"
    median_score = float(cv_results[score_column].median()) if score_column in cv_results else best_score
    best_row = (
        cv_results.sort_values(score_column, ascending=False).iloc[0]
        if score_column in cv_results and not cv_results.empty
        else None
    )

    explanations = [
        f"The selected setting achieved the highest average cross-validation score of {best_score:.4f}.",
        f"It improved on the median tested setting by {best_score - median_score:.4f}.",
    ]

    if best_row is not None:
        ranked_position = int(best_row.get("rank_test_score", 1))
        explanations.append(f"It finished at rank #{ranked_position} across the tested parameter combinations.")

    if tuning_results.get("best_parameters"):
        parameter_summary = ", ".join(
            f"{name.replace('model__', '')}={value}"
            for name, value in tuning_results["best_parameters"].items()
        )
        explanations.append(f"Best parameter set: {parameter_summary}.")

    return explanations
