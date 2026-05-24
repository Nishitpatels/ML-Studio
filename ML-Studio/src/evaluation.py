"""Evaluation helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)


def generate_confusion_matrix(y_true, y_pred):
    labels = sorted(pd.Series(y_true).dropna().unique().tolist())
    return confusion_matrix(y_true, y_pred, labels=labels), labels


def generate_classification_report(y_true, y_pred):
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return pd.DataFrame(report).transpose()


def _ensure_binary_target(y_true):
    labels = pd.Series(y_true).dropna().unique()
    if len(labels) != 2:
        raise ValueError("Curve analysis is available only for binary classification.")


def generate_roc_curve(y_true, y_probabilities):
    _ensure_binary_target(y_true)
    fpr, tpr, _ = roc_curve(y_true, y_probabilities)
    return {"fpr": fpr, "tpr": tpr, "auc": auc(fpr, tpr)}


def create_roc_chart(roc_results):
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=roc_results["fpr"], y=roc_results["tpr"], mode="lines", name=f"AUC = {roc_results['auc']:.4f}"))
    figure.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash"), name="Random"))
    figure.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", height=600)
    return figure


def generate_precision_recall_curve(y_true, y_probabilities):
    _ensure_binary_target(y_true)
    precision, recall, _ = precision_recall_curve(y_true, y_probabilities)
    return {"precision": precision, "recall": recall}


def create_precision_recall_chart(pr_results):
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=pr_results["recall"], y=pr_results["precision"], mode="lines"))
    figure.update_layout(title="Precision Recall Curve", xaxis_title="Recall", yaxis_title="Precision", height=600)
    return figure


def generate_residual_analysis(y_true, y_pred):
    y_true_series = pd.Series(y_true).reset_index(drop=True)
    y_pred_series = pd.Series(y_pred)
    return pd.DataFrame(
        {
            "Actual": y_true_series,
            "Predicted": y_pred_series,
            "Residual": y_true_series - y_pred_series,
        }
    )


def create_residual_plot(residual_dataframe):
    figure = px.scatter(residual_dataframe, x="Predicted", y="Residual", title="Residual Plot")
    figure.update_layout(height=600)
    return figure


def create_actual_vs_predicted_plot(y_true, y_pred):
    dataframe = pd.DataFrame({"Actual": y_true, "Predicted": y_pred})
    figure = px.scatter(dataframe, x="Actual", y="Predicted", title="Actual vs Predicted")
    figure.update_layout(height=600)
    return figure
