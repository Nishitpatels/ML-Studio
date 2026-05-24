"""Model export helpers."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime

import joblib


def create_export_directories():
    for directory in ("models", "models/exported", "models/metadata"):
        os.makedirs(directory, exist_ok=True)


def generate_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _safe_name(model_name: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name.strip())
    return sanitized or "model"


def export_model_pipeline(pipeline, model_name):
    create_export_directories()
    export_path = os.path.join("models", "exported", f"{_safe_name(model_name)}_{generate_timestamp()}.pkl")
    joblib.dump(pipeline, export_path)
    return export_path


def load_exported_model(model_path):
    return joblib.load(model_path)


def generate_model_metadata(model_name, metrics, problem_type, dataset_shape, feature_names=None):
    return {
        "model_name": model_name,
        "problem_type": problem_type,
        "dataset_rows": int(dataset_shape[0]),
        "dataset_columns": int(dataset_shape[1]),
        "feature_names": list(feature_names or []),
        "metrics": metrics,
        "export_timestamp": datetime.now().isoformat(timespec="seconds"),
    }


def save_model_metadata(metadata, model_name):
    create_export_directories()
    metadata_path = os.path.join("models", "metadata", f"{_safe_name(model_name)}_{generate_timestamp()}.json")
    with open(metadata_path, "w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=4)
    return metadata_path


def generate_export_summary(export_path, metadata_path):
    return {
        "Model Exported": True,
        "Model Path": export_path,
        "Metadata Path": metadata_path,
    }
