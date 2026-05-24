"""Shared Streamlit session-state contracts for ML Studio."""

from __future__ import annotations

from collections.abc import Mapping as AbcMapping
from collections.abc import MutableMapping as AbcMutableMapping
from copy import deepcopy
from hashlib import sha256
from typing import Any, Iterable, Mapping, MutableMapping

import pandas as pd


DEFAULT_SESSION_STATE: Mapping[str, Any] = {
    "dataset": None,
    "working_dataset": None,
    "transformed_dataframe": None,
    "engineered_dataset": None,
    "workflow_stage_state": {},
    "target_column": None,
    "problem_type": None,
    "current_dataset_name": None,
    "dataset_fingerprint": None,
    "trained_models": {},
    "best_model": None,
    "metrics": {},
    "feature_columns": [],
    "preprocessor": None,
    "preprocessing_config": None,
    "pipeline": None,
    "preprocessing_results": None,
    "training_results": None,
    "tuning_results": None,
    "prediction_results": None,
    "export_summary": None,
    "drift_results": None,
    "experiment_history": [],
    "chat_history": [],
    "eda_completed": False,
    "preprocessing_completed": False,
    "training_completed": False,
    "feature_engineering_completed": False,
    "model_trained": False,
}

DOWNSTREAM_KEYS = (
    "working_dataset",
    "transformed_dataframe",
    "engineered_dataset",
    "problem_type",
    "trained_models",
    "best_model",
    "metrics",
    "feature_columns",
    "preprocessor",
    "preprocessing_config",
    "pipeline",
    "preprocessing_results",
    "training_results",
    "tuning_results",
    "prediction_results",
    "export_summary",
    "drift_results",
    "experiment_history",
    "chat_history",
    "eda_completed",
    "preprocessing_completed",
    "training_completed",
    "feature_engineering_completed",
    "model_trained",
)

MODEL_ARTIFACT_KEYS = (
    "trained_models",
    "best_model",
    "metrics",
    "pipeline",
    "training_results",
    "tuning_results",
    "prediction_results",
    "export_summary",
    "drift_results",
    "experiment_history",
    "training_completed",
    "model_trained",
)

MODEL_OUTPUT_KEYS = (
    "prediction_results",
    "export_summary",
    "drift_results",
)

WORKFLOW_STAGE_KEYS = (
    "preprocessing",
    "feature_engineering",
)


def initialize_session_state(session_state: MutableMapping[str, Any]) -> None:
    """Ensure all expected keys exist with isolated mutable defaults."""

    for key, value in DEFAULT_SESSION_STATE.items():
        if key not in session_state:
            session_state[key] = deepcopy(value)

    workflow_stage_state = session_state.get("workflow_stage_state")
    if not isinstance(workflow_stage_state, AbcMutableMapping):
        session_state["workflow_stage_state"] = {}


def _copy_dataframe(dataframe: pd.DataFrame | None) -> pd.DataFrame | None:
    return dataframe.copy() if dataframe is not None else None


def _default_stage_state() -> dict[str, Any]:
    return {
        "original_dataset": None,
        "previous_stage_dataset": None,
        "current_dataset": None,
        "source_fingerprint": None,
        "current_fingerprint": None,
        "current_payload": None,
        "history": [],
    }


def _get_stage_state(
    session_state: MutableMapping[str, Any],
    stage_name: str,
) -> dict[str, Any]:
    initialize_session_state(session_state)
    workflow_stage_state = session_state["workflow_stage_state"]
    if stage_name not in workflow_stage_state:
        workflow_stage_state[stage_name] = _default_stage_state()
    return workflow_stage_state[stage_name]


def dataframe_fingerprint(dataframe: pd.DataFrame) -> str:
    """Create a stable fingerprint used to detect dataset switches."""

    dataframe_hash = pd.util.hash_pandas_object(
        dataframe.fillna("<NA>").astype(str),
        index=True,
    ).values.tobytes()
    metadata = f"{tuple(dataframe.columns)}|{tuple(map(str, dataframe.dtypes))}|{dataframe.shape}"
    return sha256(metadata.encode("utf-8") + dataframe_hash).hexdigest()


def clear_downstream_state(
    session_state: MutableMapping[str, Any],
    *,
    keep_working_dataset: bool = False,
) -> None:
    """Reset artifacts that become stale after a dataset or target change."""

    for key in DOWNSTREAM_KEYS:
        if keep_working_dataset and key in {
            "working_dataset",
            "transformed_dataframe",
            "engineered_dataset",
            "preprocessing_config",
        }:
            continue
        session_state[key] = deepcopy(DEFAULT_SESSION_STATE[key])


def clear_model_artifacts(session_state: MutableMapping[str, Any]) -> None:
    """Reset model artifacts that become stale after preprocessing changes."""

    for key in MODEL_ARTIFACT_KEYS:
        session_state[key] = deepcopy(DEFAULT_SESSION_STATE[key])


def clear_model_outputs(session_state: MutableMapping[str, Any]) -> None:
    """Reset outputs that belong to a previously active model."""

    for key in MODEL_OUTPUT_KEYS:
        session_state[key] = deepcopy(DEFAULT_SESSION_STATE[key])


def set_preprocessing_results(
    session_state: MutableMapping[str, Any],
    preprocessing_results: dict[str, Any],
    preprocessing_config: dict[str, Any] | None = None,
) -> None:
    """Store preprocessing artifacts and invalidate stale model artifacts."""

    initialize_session_state(session_state)
    clear_model_artifacts(session_state)
    feature_schema = preprocessing_results.get("feature_schema")
    if feature_schema is None and preprocessing_results.get("X_train") is not None:
        feature_schema = preprocessing_results["X_train"].columns
    session_state["preprocessing_results"] = preprocessing_results
    session_state["preprocessor"] = preprocessing_results.get("preprocessor")
    if preprocessing_config is not None:
        session_state["preprocessing_config"] = preprocessing_config
    transformed_dataframe = preprocessing_results.get("transformed_dataframe")
    if transformed_dataframe is not None:
        session_state["transformed_dataframe"] = transformed_dataframe.copy()
    session_state["feature_columns"] = list(feature_schema) if feature_schema is not None else []
    session_state["preprocessing_completed"] = True


def set_training_results(
    session_state: MutableMapping[str, Any],
    training_results: dict[str, Any],
) -> None:
    """Store model-training artifacts under the canonical workflow keys."""

    initialize_session_state(session_state)
    clear_model_outputs(session_state)

    best_model = training_results.get("best_model")
    session_state["training_results"] = training_results
    session_state["trained_models"] = training_results.get("trained_models", {})
    session_state["best_model"] = best_model
    session_state["metrics"] = best_model.get("metrics", {}) if best_model else {}
    session_state["pipeline"] = best_model.get("pipeline") if best_model else None
    session_state["training_completed"] = best_model is not None
    session_state["model_trained"] = best_model is not None


def set_tuning_results(
    session_state: MutableMapping[str, Any],
    tuning_results: dict[str, Any],
    training_results: dict[str, Any] | None = None,
) -> None:
    """Store tuning artifacts and optionally promote the tuned model."""

    initialize_session_state(session_state)
    session_state["tuning_results"] = tuning_results
    if training_results is not None:
        set_training_results(session_state, training_results)
        session_state["tuning_results"] = tuning_results


def set_dataset(
    session_state: MutableMapping[str, Any],
    dataframe: pd.DataFrame,
    dataset_name: str | None,
) -> bool:
    """Store a dataset and invalidate stale artifacts only when it changed."""

    initialize_session_state(session_state)
    fingerprint = dataframe_fingerprint(dataframe)
    changed = fingerprint != session_state.get("dataset_fingerprint")

    session_state["dataset"] = dataframe.copy()
    session_state["current_dataset_name"] = dataset_name
    session_state["dataset_fingerprint"] = fingerprint

    if changed:
        if session_state.get("target_column") not in dataframe.columns:
            session_state["target_column"] = None
        clear_downstream_state(session_state)
        session_state["working_dataset"] = dataframe.copy()
        session_state["workflow_stage_state"] = {}
    elif session_state.get("working_dataset") is None:
        session_state["working_dataset"] = dataframe.copy()

    return changed


def set_target_column(
    session_state: MutableMapping[str, Any],
    target_column: str | None,
) -> bool:
    """Update the target and clear model artifacts if the target changed."""

    initialize_session_state(session_state)
    changed = target_column != session_state.get("target_column")
    session_state["target_column"] = target_column

    if changed:
        clear_downstream_state(session_state)
        dataset = session_state.get("dataset")
        session_state["working_dataset"] = dataset.copy() if dataset is not None else None
        session_state["workflow_stage_state"] = {}
    elif session_state.get("working_dataset") is None:
        dataset = session_state.get("dataset")
        session_state["working_dataset"] = dataset.copy() if dataset is not None else None

    return changed


def set_working_dataset(
    session_state: MutableMapping[str, Any],
    dataframe: pd.DataFrame,
    *,
    engineered: bool = False,
    preserve_engineered_dataset: bool = False,
) -> None:
    """Store the active modeling dataset and invalidate stale artifacts."""

    initialize_session_state(session_state)
    session_state["working_dataset"] = dataframe.copy()
    if engineered:
        session_state["engineered_dataset"] = dataframe.copy()
    else:
        session_state["transformed_dataframe"] = dataframe.copy()
        if not preserve_engineered_dataset:
            session_state["engineered_dataset"] = None
    clear_downstream_state(session_state, keep_working_dataset=True)
    session_state["feature_engineering_completed"] = engineered


def get_modeling_dataset(session_state: Mapping[str, Any]) -> pd.DataFrame | None:
    """Return the active dataset used by preprocessing/training."""

    working_dataset = session_state.get("working_dataset")
    if working_dataset is not None:
        return working_dataset
    return session_state.get("dataset")


def get_engineered_data(session_state: Mapping[str, Any]) -> pd.DataFrame | None:
    """Return the latest transformed dataset for cross-page inspection."""

    transformed_dataframe = session_state.get("transformed_dataframe")
    if transformed_dataframe is not None:
        return transformed_dataframe.copy()

    engineered_dataset = session_state.get("engineered_dataset")
    if engineered_dataset is not None:
        return engineered_dataset.copy()

    working_dataset = session_state.get("working_dataset")
    if working_dataset is not None:
        return working_dataset.copy()

    dataset = session_state.get("dataset")
    return dataset.copy() if dataset is not None else None


def get_engineered_data_stage_label(session_state: Mapping[str, Any]) -> str:
    """Describe which workflow stage produced the current engineered data."""

    if session_state.get("transformed_dataframe") is not None:
        return "Preprocessed Dataset"
    if session_state.get("engineered_dataset") is not None:
        return "Feature-Engineered Dataset"
    if session_state.get("working_dataset") is not None:
        return "Active Working Dataset"
    return "Uploaded Dataset"


def sync_workflow_stage(
    session_state: MutableMapping[str, Any],
    stage_name: str,
    source_dataframe: pd.DataFrame,
) -> dict[str, Any]:
    """Keep per-stage snapshots aligned to the current upstream dataset."""

    if stage_name not in WORKFLOW_STAGE_KEYS:
        raise ValueError(f"Unsupported workflow stage: {stage_name}")

    stage_state = _get_stage_state(session_state, stage_name)
    source_fingerprint = dataframe_fingerprint(source_dataframe)
    original_dataset = session_state.get("dataset")
    copied_original_dataset = _copy_dataframe(original_dataset)

    if stage_state["source_fingerprint"] != source_fingerprint:
        stage_state["original_dataset"] = (
            copied_original_dataset if copied_original_dataset is not None else source_dataframe.copy()
        )
        stage_state["previous_stage_dataset"] = source_dataframe.copy()
        stage_state["current_dataset"] = source_dataframe.copy()
        stage_state["source_fingerprint"] = source_fingerprint
        stage_state["current_fingerprint"] = source_fingerprint
        stage_state["current_payload"] = None
        stage_state["history"] = []
    elif stage_state["current_dataset"] is None:
        stage_state["current_dataset"] = source_dataframe.copy()
        stage_state["current_fingerprint"] = source_fingerprint

    return stage_state


def get_workflow_stage_dataset(
    session_state: Mapping[str, Any],
    stage_name: str,
    *,
    snapshot_name: str = "current_dataset",
) -> pd.DataFrame | None:
    """Return a stage snapshot when available."""

    workflow_stage_state = session_state.get("workflow_stage_state", {})
    stage_state = workflow_stage_state.get(stage_name, {})
    dataframe = stage_state.get(snapshot_name)
    return _copy_dataframe(dataframe)


def get_workflow_stage_payload(
    session_state: Mapping[str, Any],
    stage_name: str,
) -> Any:
    """Return the current payload stored for a workflow stage."""

    workflow_stage_state = session_state.get("workflow_stage_state", {})
    stage_state = workflow_stage_state.get(stage_name, {})
    return deepcopy(stage_state.get("current_payload"))


def apply_workflow_stage_result(
    session_state: MutableMapping[str, Any],
    stage_name: str,
    dataframe: pd.DataFrame,
    payload: Any = None,
) -> None:
    """Persist the current stage output and keep a reversible history."""

    stage_state = _get_stage_state(session_state, stage_name)
    current_dataset = stage_state.get("current_dataset")

    if current_dataset is not None:
        current_fingerprint = dataframe_fingerprint(current_dataset)
        next_fingerprint = dataframe_fingerprint(dataframe)
        if current_fingerprint != next_fingerprint:
            stage_state["history"].append(
                {
                    "dataset": current_dataset.copy(),
                    "payload": deepcopy(stage_state.get("current_payload")),
                }
            )

    stage_state["current_dataset"] = dataframe.copy()
    stage_state["current_fingerprint"] = dataframe_fingerprint(dataframe)
    stage_state["current_payload"] = deepcopy(payload)


def undo_workflow_stage(
    session_state: MutableMapping[str, Any],
    stage_name: str,
) -> tuple[pd.DataFrame | None, Any]:
    """Restore the previous stage-local snapshot if available."""

    stage_state = _get_stage_state(session_state, stage_name)
    history = stage_state.get("history", [])

    if history:
        snapshot = history.pop()
        stage_state["current_dataset"] = _copy_dataframe(snapshot.get("dataset"))
        stage_state["current_payload"] = deepcopy(snapshot.get("payload"))
        current_dataset = stage_state.get("current_dataset")
        stage_state["current_fingerprint"] = (
            dataframe_fingerprint(current_dataset) if current_dataset is not None else None
        )
        return _copy_dataframe(current_dataset), deepcopy(stage_state.get("current_payload"))

    previous_dataset = stage_state.get("previous_stage_dataset")
    stage_state["current_dataset"] = _copy_dataframe(previous_dataset)
    stage_state["current_payload"] = None
    stage_state["current_fingerprint"] = (
        dataframe_fingerprint(previous_dataset) if previous_dataset is not None else None
    )
    return _copy_dataframe(previous_dataset), None


def reset_workflow_stage(
    session_state: MutableMapping[str, Any],
    stage_name: str,
) -> tuple[pd.DataFrame | None, Any]:
    """Clear page-local history and restore the incoming dataset for that stage."""

    stage_state = _get_stage_state(session_state, stage_name)
    previous_dataset = stage_state.get("previous_stage_dataset")
    stage_state["history"] = []
    stage_state["current_dataset"] = _copy_dataframe(previous_dataset)
    stage_state["current_payload"] = None
    stage_state["current_fingerprint"] = (
        dataframe_fingerprint(previous_dataset) if previous_dataset is not None else None
    )
    return _copy_dataframe(previous_dataset), None


def get_missing_or_empty_keys(
    session_state: Mapping[str, Any],
    required_keys: Iterable[str],
) -> list[str]:
    """Return required keys that are absent or currently empty."""

    return [
        key
        for key in required_keys
        if key not in session_state or is_missing_or_empty(session_state.get(key))
    ]


def is_missing_or_empty(value: Any) -> bool:
    """Return True for empty workflow artifacts while avoiding dataframe truthiness."""

    if value is None:
        return True
    if isinstance(value, (str, bytes)):
        return len(value) == 0
    if isinstance(value, AbcMapping):
        return len(value) == 0
    if isinstance(value, (list, tuple, set)):
        return len(value) == 0
    return False
