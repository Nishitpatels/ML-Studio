"""Model registry and problem-type detection."""

from __future__ import annotations

from typing import Any

import pandas as pd
from pandas.api.types import is_bool_dtype, is_integer_dtype, is_numeric_dtype
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


def detect_problem_type(target_series: pd.Series) -> str:
    """Infer classification vs regression using dtype plus cardinality."""

    non_null = target_series.dropna()
    if non_null.empty:
        raise ValueError("Target column contains no usable values.")

    unique_values = non_null.nunique()
    unique_ratio = unique_values / len(non_null)

    if is_bool_dtype(non_null) or not is_numeric_dtype(non_null):
        return "classification"
    if is_integer_dtype(non_null) and unique_values <= 20:
        return "classification"
    if unique_values <= 20 and unique_ratio <= 0.05:
        return "classification"
    return "regression"


def get_classification_models() -> dict[str, Any]:
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "Extra Trees": ExtraTreesClassifier(n_estimators=100, random_state=42),
        "KNN": KNeighborsClassifier(),
        "SVM": SVC(probability=True, random_state=42),
    }


def get_regression_models() -> dict[str, Any]:
    return {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(),
        "Lasso Regression": Lasso(),
        "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=42),
        "Extra Trees Regressor": ExtraTreesRegressor(n_estimators=100, random_state=42),
        "KNN Regressor": KNeighborsRegressor(),
        "SVR": SVR(),
    }


def get_models_by_problem_type(problem_type: str) -> dict[str, Any]:
    if problem_type == "classification":
        return get_classification_models()
    if problem_type == "regression":
        return get_regression_models()
    raise ValueError(f"Unsupported problem type: {problem_type}")


def get_model_names(problem_type: str) -> list[str]:
    return list(get_models_by_problem_type(problem_type))


def get_single_model(model_name: str, problem_type: str):
    models = get_models_by_problem_type(problem_type)
    if model_name not in models:
        raise ValueError(f"Model '{model_name}' not found.")
    return models[model_name]


def generate_model_recommendations(dataframe: pd.DataFrame, target_column: str) -> list[str]:
    problem_type = detect_problem_type(dataframe[target_column])
    recommendations: list[str] = []
    if len(dataframe) < 1000:
        recommendations.append("Smaller dataset detected; start with simpler baselines plus tree ensembles.")
    elif len(dataframe) > 100000:
        recommendations.append("Large dataset detected; prefer scalable models and tighter validation loops.")
    categorical_count = len(dataframe.select_dtypes(include=["object", "category", "bool"]).columns)
    numerical_count = len(dataframe.select_dtypes(include="number").columns)
    if categorical_count > numerical_count:
        recommendations.append("Categorical-heavy feature space detected; tree ensembles are a strong first pass.")
    recommendations.append(
        "Classification task detected." if problem_type == "classification" else "Regression task detected."
    )
    return recommendations


def get_model_information() -> dict[str, dict[str, Any]]:
    return {
        "Logistic Regression": {
            "description": "Strong linear classification baseline.",
            "when_to_use": "Good first model when you want a fast, stable benchmark.",
            "strengths": [
                "Easy to train and explain",
                "Works well on linearly separable patterns",
                "Usually a strong baseline for binary or multiclass tasks",
            ],
            "watch_outs": [
                "Misses complex non-linear relationships",
                "Can be sensitive to extreme outliers",
            ],
        },
        "Decision Tree": {
            "description": "Single tree classifier with readable splits.",
            "when_to_use": "Useful when interpretability matters more than peak accuracy.",
            "strengths": [
                "Very easy to explain",
                "Handles non-linear rules",
                "Works with mixed feature types after preprocessing",
            ],
            "watch_outs": [
                "Can overfit quickly",
                "Performance may vary with small data changes",
            ],
        },
        "Random Forest": {
            "description": "Robust tree ensemble for classification.",
            "when_to_use": "A strong default when you need dependable performance on mixed datasets.",
            "strengths": [
                "Handles non-linearity well",
                "Robust against overfitting compared with a single tree",
                "Works well with many real-world tabular datasets",
            ],
            "watch_outs": [
                "Less interpretable than a single tree",
                "Can be slower than simpler linear models",
            ],
        },
        "Gradient Boosting": {
            "description": "Boosted tree classifier that learns stage by stage.",
            "when_to_use": "Useful when you want stronger tabular performance and can spend more time tuning.",
            "strengths": [
                "Captures complex relationships",
                "Often strong on structured data",
                "Can outperform bagged trees after tuning",
            ],
            "watch_outs": [
                "More sensitive to tuning choices",
                "Can overfit if pushed too far",
            ],
        },
        "Extra Trees": {
            "description": "Highly randomized tree ensemble classifier.",
            "when_to_use": "Helpful when you want a fast ensemble with strong variance reduction.",
            "strengths": [
                "Usually fast to train",
                "Handles non-linear patterns",
                "Often competitive with random forests",
            ],
            "watch_outs": [
                "Less intuitive than a single tree",
                "May underperform if randomness is too aggressive for the dataset",
            ],
        },
        "KNN": {
            "description": "Distance-based classifier using nearby examples.",
            "when_to_use": "Useful for smaller datasets where local similarity matters.",
            "strengths": [
                "Simple intuition",
                "Can model complex local boundaries",
                "No heavy training phase",
            ],
            "watch_outs": [
                "Prediction can be slow on large datasets",
                "Sensitive to scaling and noisy features",
            ],
        },
        "SVM": {
            "description": "Margin-based classifier that can use kernels.",
            "when_to_use": "Helpful for medium-sized datasets with cleaner boundaries.",
            "strengths": [
                "Can perform well on complex boundaries",
                "Effective in high-dimensional spaces",
                "Strong margin-based decision making",
            ],
            "watch_outs": [
                "Can be slow on larger datasets",
                "Less transparent for beginners",
            ],
        },
        "Linear Regression": {
            "description": "Fast linear regression baseline.",
            "when_to_use": "Start here when you want a simple regression benchmark.",
            "strengths": [
                "Easy to explain",
                "Fast to train",
                "Useful baseline for continuous targets",
            ],
            "watch_outs": [
                "Cannot capture non-linear patterns by itself",
                "Sensitive to outliers and multicollinearity",
            ],
        },
        "Ridge Regression": {
            "description": "Linear regression with L2 regularization.",
            "when_to_use": "A good choice when linear features are useful but coefficients need stabilization.",
            "strengths": [
                "Handles correlated features better than plain linear regression",
                "Stable baseline for many regression tasks",
                "Usually low variance",
            ],
            "watch_outs": [
                "Still limited to mostly linear relationships",
                "May underfit more complex patterns",
            ],
        },
        "Lasso Regression": {
            "description": "Linear regression with L1 regularization.",
            "when_to_use": "Useful when you want a sparse linear model that can downweight weak features.",
            "strengths": [
                "Can simplify the model by shrinking weak coefficients to zero",
                "Fast and interpretable",
                "Helpful when feature selection matters",
            ],
            "watch_outs": [
                "Can underfit if regularization is too strong",
                "Still mainly a linear model",
            ],
        },
        "Decision Tree Regressor": {
            "description": "Single tree regressor with readable rules.",
            "when_to_use": "Useful for explainable regression baselines with non-linear behavior.",
            "strengths": [
                "Handles non-linear splits",
                "Easy to visualize",
                "Minimal assumptions about feature relationships",
            ],
            "watch_outs": [
                "Can overfit quickly",
                "Predictions can be unstable across splits",
            ],
        },
        "Random Forest Regressor": {
            "description": "Robust bagged tree ensemble for regression.",
            "when_to_use": "Strong default for tabular regression with mixed signal types.",
            "strengths": [
                "Handles non-linear patterns well",
                "Robust against overfitting",
                "Usually strong on practical tabular problems",
            ],
            "watch_outs": [
                "Less interpretable than simpler regressors",
                "Can miss smooth extrapolation outside the training range",
            ],
        },
        "Gradient Boosting Regressor": {
            "description": "Boosted tree regressor built in stages.",
            "when_to_use": "Good when you want stronger performance and are willing to tune carefully.",
            "strengths": [
                "Captures subtle tabular patterns",
                "Often strong predictive performance",
                "Can beat bagged trees after tuning",
            ],
            "watch_outs": [
                "More tuning-sensitive",
                "Training can take longer than simpler ensembles",
            ],
        },
        "Extra Trees Regressor": {
            "description": "Highly randomized tree ensemble regressor.",
            "when_to_use": "Helpful when you want a fast, strong tree ensemble for regression.",
            "strengths": [
                "Fast ensemble training",
                "Captures non-linear signal",
                "Often competitive with random forests",
            ],
            "watch_outs": [
                "Less interpretable than simpler baselines",
                "Randomness can make it less stable on some datasets",
            ],
        },
        "KNN Regressor": {
            "description": "Distance-based regressor using nearby points.",
            "when_to_use": "Useful on smaller datasets where local neighborhoods are informative.",
            "strengths": [
                "Simple intuition",
                "Can model local non-linear behavior",
                "No heavy training stage",
            ],
            "watch_outs": [
                "Prediction slows down on larger datasets",
                "Sensitive to scaling and noisy dimensions",
            ],
        },
        "SVR": {
            "description": "Kernel-based support vector regression.",
            "when_to_use": "Helpful for medium-sized regression tasks with complex boundaries.",
            "strengths": [
                "Handles non-linear relationships",
                "Can work well in high-dimensional spaces",
                "Strong margin-based formulation",
            ],
            "watch_outs": [
                "Can be slow on larger datasets",
                "Less transparent for beginners",
            ],
        },
    }


def perform_model_analysis(dataframe: pd.DataFrame, target_column: str) -> dict[str, Any]:
    problem_type = detect_problem_type(dataframe[target_column])
    models = get_models_by_problem_type(problem_type)
    model_information = get_model_information()
    return {
        "problem_type": problem_type,
        "available_models": list(models),
        "total_models": len(models),
        "recommendations": generate_model_recommendations(dataframe, target_column),
        "model_information": {
            model_name: model_information[model_name]
            for model_name in models
            if model_name in model_information
        },
    }
