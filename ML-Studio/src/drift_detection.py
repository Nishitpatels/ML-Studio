"""Simple PSI-based drift detection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px


def calculate_psi(expected_series, actual_series, bins: int = 10) -> float:
    expected = pd.to_numeric(expected_series, errors="coerce").dropna()
    actual = pd.to_numeric(actual_series, errors="coerce").dropna()
    if expected.empty or actual.empty:
        raise ValueError("PSI requires non-empty numeric samples.")
    quantiles = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(quantiles) < 2:
        raise ValueError("PSI requires at least two distinct reference values.")
    expected_counts, bin_edges = np.histogram(expected, bins=quantiles)
    actual_counts, _ = np.histogram(actual, bins=bin_edges)
    expected_distribution = np.clip(expected_counts / max(expected_counts.sum(), 1), 1e-6, None)
    actual_distribution = np.clip(actual_counts / max(actual_counts.sum(), 1), 1e-6, None)
    return round(float(np.sum((expected_distribution - actual_distribution) * np.log(expected_distribution / actual_distribution))), 4)


def interpret_psi(psi_value):
    if psi_value < 0.1:
        return "No Drift"
    if psi_value < 0.25:
        return "Moderate Drift"
    return "Significant Drift"


def detect_feature_drift(reference_dataframe, current_dataframe):
    common_numeric_columns = [
        column
        for column in reference_dataframe.select_dtypes(include="number").columns
        if column in current_dataframe.columns
    ]
    rows = []
    for column in common_numeric_columns:
        try:
            psi_value = calculate_psi(reference_dataframe[column], current_dataframe[column])
            rows.append({"Feature": column, "PSI": psi_value, "Drift Status": interpret_psi(psi_value)})
        except Exception:
            continue
    return pd.DataFrame(rows, columns=["Feature", "PSI", "Drift Status"])


def create_drift_chart(drift_dataframe):
    if drift_dataframe.empty:
        return px.bar(title="Feature Drift Analysis")
    figure = px.bar(drift_dataframe, x="Feature", y="PSI", color="Drift Status", title="Feature Drift Analysis")
    figure.update_layout(height=600)
    return figure


def generate_drift_summary(drift_dataframe):
    if drift_dataframe.empty:
        return {
            "Total Features": 0,
            "No Drift Features": 0,
            "Moderate Drift Features": 0,
            "Significant Drift Features": 0,
        }
    return {
        "Total Features": len(drift_dataframe),
        "No Drift Features": int((drift_dataframe["Drift Status"] == "No Drift").sum()),
        "Moderate Drift Features": int((drift_dataframe["Drift Status"] == "Moderate Drift").sum()),
        "Significant Drift Features": int((drift_dataframe["Drift Status"] == "Significant Drift").sum()),
    }
