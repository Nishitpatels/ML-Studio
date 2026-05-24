# =========================================================
# FILE: src/visualization.py
# =========================================================
# Responsibilities:
# 1. Missing value visualizations
# 2. Correlation heatmaps
# 3. Numerical feature distributions
# 4. Categorical feature visualizations
# 5. Outlier visualizations
# 6. Target distribution analysis
# 7. Interactive Plotly charts
# =========================================================

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde


# ---------------------------------------------------------
# MISSING VALUE BAR CHART
# ---------------------------------------------------------
def plot_missing_values(dataframe: pd.DataFrame):
    """
    Creates missing value visualization.
    """

    missing_values = dataframe.isnull().sum()

    missing_values = missing_values[
        missing_values > 0
    ].sort_values(ascending=False)

    if len(missing_values) == 0:
        return None

    figure = px.bar(
        x=missing_values.index,
        y=missing_values.values,
        title="Missing Value Analysis",
        labels={
            "x": "Columns",
            "y": "Missing Values"
        }
    )

    figure.update_layout(
        xaxis_tickangle=-45,
        height=500
    )

    return figure


# ---------------------------------------------------------
# CORRELATION HEATMAP
# ---------------------------------------------------------
def plot_correlation_heatmap(dataframe: pd.DataFrame):
    """
    Creates correlation heatmap for numerical features.
    """

    numerical_dataframe = dataframe.select_dtypes(
        include=["int64", "float64"]
    )

    if numerical_dataframe.shape[1] < 2:
        return None

    correlation_matrix = numerical_dataframe.corr()

    figure = px.imshow(
        correlation_matrix,
        text_auto=True,
        aspect="auto",
        title="Correlation Heatmap"
    )

    figure.update_layout(
        height=700
    )

    return figure


# ---------------------------------------------------------
# FEATURE DISTRIBUTION PLOT
# ---------------------------------------------------------
def plot_feature_distribution(
    dataframe: pd.DataFrame,
    column: str
):
    """
    Creates histogram distribution plot.
    """

    figure = px.histogram(
        dataframe,
        x=column,
        nbins=30,
        title=f"Distribution of {column}"
    )

    figure.update_layout(
        height=500
    )

    return figure


def plot_kde_distribution(
    dataframe: pd.DataFrame,
    columns: list[str],
):
    """
    Creates KDE/PDF distribution plot for selected numerical columns.
    """

    figure = go.Figure()

    for column in columns:
        series = pd.to_numeric(
            dataframe[column],
            errors="coerce"
        ).dropna()

        if series.nunique() < 2:
            continue

        x_axis = np.linspace(
            series.min(),
            series.max(),
            200
        )

        figure.add_trace(
            go.Scatter(
                x=x_axis,
                y=gaussian_kde(series)(x_axis),
                mode="lines",
                name=column
            )
        )

    if len(figure.data) == 0:
        return None

    figure.update_layout(
        title="Probability Density (KDE) Plot",
        xaxis_title="Value",
        yaxis_title="Density",
        height=500
    )

    return figure


def plot_distribution_with_kde(
    dataframe: pd.DataFrame,
    column: str,
):
    """
    Creates a histogram with an overlaid KDE/PDF line for one numerical feature.
    """

    series = pd.to_numeric(
        dataframe[column],
        errors="coerce"
    ).dropna()

    if series.nunique() < 2:
        return None

    figure = go.Figure()
    figure.add_trace(
        go.Histogram(
            x=series,
            nbinsx=30,
            histnorm="probability density",
            name="Histogram",
            marker_color="#4C78A8",
            opacity=0.75,
        )
    )

    x_axis = np.linspace(
        series.min(),
        series.max(),
        200
    )
    figure.add_trace(
        go.Scatter(
            x=x_axis,
            y=gaussian_kde(series)(x_axis),
            mode="lines",
            name="KDE / PDF",
            line=dict(color="#2E8B57", width=3),
        )
    )

    figure.update_layout(
        title=f"Distribution of {column}",
        xaxis_title=column,
        yaxis_title="Density",
        barmode="overlay",
        height=460,
        legend_title_text="",
    )

    return figure


# ---------------------------------------------------------
# BOXPLOT VISUALIZATION
# ---------------------------------------------------------
def plot_boxplot(
    dataframe: pd.DataFrame,
    column: str
):
    """
    Creates boxplot for outlier analysis.
    """

    figure = px.box(
        dataframe,
        y=column,
        title=f"Boxplot of {column}"
    )

    figure.update_layout(
        height=500
    )

    return figure


# ---------------------------------------------------------
# CATEGORICAL DISTRIBUTION
# ---------------------------------------------------------
def plot_categorical_distribution(
    dataframe: pd.DataFrame,
    column: str
):
    """
    Creates categorical distribution chart.
    """

    value_counts = dataframe[column].value_counts()

    figure = px.bar(
        x=value_counts.index.astype(str),
        y=value_counts.values,
        title=f"Categorical Distribution of {column}",
        labels={
            "x": column,
            "y": "Count"
        }
    )

    figure.update_layout(
        xaxis_tickangle=-45,
        height=500
    )

    return figure


# ---------------------------------------------------------
# TARGET DISTRIBUTION
# ---------------------------------------------------------
def plot_target_distribution(
    dataframe: pd.DataFrame,
    target_column: str
):
    """
    Creates target distribution visualization.
    """

    unique_values = dataframe[target_column].nunique()

    # Classification
    if unique_values <= 20:

        value_counts = dataframe[
            target_column
        ].value_counts()

        figure = px.pie(
            values=value_counts.values,
            names=value_counts.index.astype(str),
            title=f"Target Distribution - {target_column}"
        )

    # Regression
    else:

        figure = px.histogram(
            dataframe,
            x=target_column,
            nbins=30,
            title=f"Target Distribution - {target_column}"
        )

    figure.update_layout(
        height=500
    )

    return figure


# ---------------------------------------------------------
# CLASS BALANCE CHART
# ---------------------------------------------------------
def plot_class_balance(
    dataframe: pd.DataFrame,
    target_column: str
):
    """
    Creates class balance visualization.
    """

    class_distribution = dataframe[
        target_column
    ].value_counts()

    figure = px.bar(
        x=class_distribution.index.astype(str),
        y=class_distribution.values,
        title="Class Balance Analysis",
        labels={
            "x": "Class",
            "y": "Count"
        }
    )

    figure.update_layout(
        height=500
    )

    return figure


# ---------------------------------------------------------
# SCATTER PLOT
# ---------------------------------------------------------
def plot_scatter(
    dataframe: pd.DataFrame,
    x_column: str,
    y_column: str,
    color_column: str = None
):
    """
    Creates scatter plot.
    """

    figure = px.scatter(
        dataframe,
        x=x_column,
        y=y_column,
        color=color_column,
        title=f"{x_column} vs {y_column}"
    )

    figure.update_layout(
        height=600
    )

    return figure


# ---------------------------------------------------------
# FEATURE IMPORTANCE PLOT
# ---------------------------------------------------------
def plot_feature_importance(
    feature_names,
    importance_values
):
    """
    Creates feature importance chart.
    """

    importance_dataframe = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance_values
    })

    importance_dataframe = importance_dataframe.sort_values(
        by="Importance",
        ascending=False
    )

    figure = px.bar(
        importance_dataframe,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Feature Importance"
    )

    figure.update_layout(
        height=700
    )

    return figure


# ---------------------------------------------------------
# NUMERICAL SUMMARY CHART
# ---------------------------------------------------------
def plot_numerical_summary(
    dataframe: pd.DataFrame
):
    """
    Creates multi-feature numerical visualization.
    """

    numerical_columns = dataframe.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    if len(numerical_columns) == 0:
        return None

    figure = make_subplots(
        rows=len(numerical_columns),
        cols=1,
        subplot_titles=numerical_columns
    )

    for index, column in enumerate(numerical_columns):

        histogram = go.Histogram(
            x=dataframe[column],
            name=column
        )

        figure.add_trace(
            histogram,
            row=index + 1,
            col=1
        )

    figure.update_layout(
        height=300 * len(numerical_columns),
        title="Numerical Feature Summary"
    )

    return figure


# ---------------------------------------------------------
# OUTLIER VISUALIZATION
# ---------------------------------------------------------
def plot_outlier_analysis(
    dataframe: pd.DataFrame,
    column: str
):
    """
    Creates outlier visualization.
    """

    figure = go.Figure()

    figure.add_trace(
        go.Box(
            y=dataframe[column],
            name=column,
            boxpoints="outliers"
        )
    )

    figure.update_layout(
        title=f"Outlier Analysis - {column}",
        height=500
    )

    return figure


# ---------------------------------------------------------
# DATASET OVERVIEW PIE CHART
# ---------------------------------------------------------
def plot_column_type_distribution(
    numerical_count: int,
    categorical_count: int,
    datetime_count: int
):
    """
    Creates column type distribution chart.
    """

    figure = px.pie(
        values=[
            numerical_count,
            categorical_count,
            datetime_count
        ],
        names=[
            "Numerical",
            "Categorical",
            "Datetime"
        ],
        title="Column Type Distribution"
    )

    figure.update_layout(
        height=500
    )

    return figure


def plot_model_metric_comparison(
    evaluation_dataframe: pd.DataFrame,
    metric: str,
    best_model_name: str | None,
    *,
    higher_is_better: bool = True
):
    """
    Creates model comparison chart with best model highlighting.
    """

    comparison_dataframe = evaluation_dataframe.dropna(
        subset=[metric]
    ).copy()

    if comparison_dataframe.empty:
        return None

    comparison_dataframe = comparison_dataframe.sort_values(
        metric,
        ascending=not higher_is_better
    )

    comparison_dataframe["Highlight"] = comparison_dataframe[
        "Model"
    ].eq(best_model_name).map(
        {
            True: "Best Model",
            False: "Other Models"
        }
    )

    figure = px.bar(
        comparison_dataframe,
        x="Model",
        y=metric,
        color="Highlight",
        color_discrete_map={
            "Best Model": "#2E8B57",
            "Other Models": "#4C78A8"
        },
        text=metric,
        title=f"{metric} Comparison"
    )

    figure.update_traces(
        texttemplate="%{text:.4f}",
        textposition="outside"
    )

    figure.update_layout(
        height=550,
        legend_title_text="",
        yaxis_title=metric
    )

    return figure


def plot_metric_bar_chart(
    metrics: dict[str, float],
    title: str
):
    """
    Creates compact metric visualization.
    """

    if not metrics:
        return None

    metric_dataframe = pd.DataFrame({
        "Metric": list(metrics.keys()),
        "Value": list(metrics.values())
    })

    figure = px.bar(
        metric_dataframe,
        x="Metric",
        y="Value",
        text="Value",
        title=title
    )

    figure.update_traces(
        texttemplate="%{text:.4f}",
        textposition="outside"
    )

    figure.update_layout(
        height=450
    )

    return figure


def plot_tuning_progression(cv_results: pd.DataFrame):
    """
    Creates a tuning progression chart across tested parameter sets.
    """

    required_columns = {"candidate_id", "mean_test_score"}
    if cv_results.empty or not required_columns.issubset(cv_results.columns):
        return None

    plot_frame = cv_results.sort_values("candidate_id").copy()
    plot_frame["Best So Far"] = plot_frame["mean_test_score"].cummax()

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=plot_frame["candidate_id"],
            y=plot_frame["mean_test_score"],
            mode="lines+markers",
            name="Candidate Score",
            line=dict(color="#4C78A8"),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=plot_frame["candidate_id"],
            y=plot_frame["Best So Far"],
            mode="lines",
            name="Best So Far",
            line=dict(color="#2E8B57", dash="dash"),
        )
    )
    figure.update_layout(
        title="Tuning Progression",
        xaxis_title="Candidate Order",
        yaxis_title="Mean CV Score",
        height=450,
    )
    return figure


def plot_parameter_impact(cv_results: pd.DataFrame, parameter_name: str):
    """
    Creates a parameter-versus-score chart for one tuned parameter.
    """

    if cv_results.empty or parameter_name not in cv_results.columns or "mean_test_score" not in cv_results.columns:
        return None

    plot_frame = cv_results[[parameter_name, "mean_test_score", "rank_test_score"]].dropna().copy()
    if plot_frame.empty:
        return None

    plot_frame["parameter_value"] = plot_frame[parameter_name].astype(str)
    figure = px.strip(
        plot_frame,
        x="parameter_value",
        y="mean_test_score",
        color="rank_test_score",
        title=f"{parameter_name.replace('param_model__', '').replace('param_', '')} vs CV Score",
        labels={
            "parameter_value": "Parameter Value",
            "mean_test_score": "Mean CV Score",
            "rank_test_score": "Rank",
        },
    )
    figure.update_layout(height=450)
    return figure
