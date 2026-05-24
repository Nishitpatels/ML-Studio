# =========================================================
# FILE: src/experiment_tracker.py
# =========================================================

import os
import json
import pandas as pd

from datetime import datetime


# =========================================================
# CREATE EXPERIMENT DIRECTORY
# =========================================================
def create_experiment_directory():

    os.makedirs(
        "experiments",
        exist_ok=True
    )


# =========================================================
# GENERATE TIMESTAMP
# =========================================================
def generate_timestamp():

    return datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )


# =========================================================
# GENERATE EXPERIMENT ID
# =========================================================
def generate_experiment_id():

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    return f"EXP_{timestamp}"


def _sanitize_for_json(value):

    if isinstance(
        value,
        (
            str,
            int,
            float,
            bool
        )
    ) or value is None:

        return value

    if isinstance(
        value,
        dict
    ):

        return {
            str(key): _sanitize_for_json(item)
            for key, item in value.items()
        }

    if isinstance(
        value,
        (
            list,
            tuple,
            set
        )
    ):

        return [
            _sanitize_for_json(item)
            for item in value
        ]

    return str(value)


# =========================================================
# SAVE EXPERIMENT
# =========================================================
def save_experiment(

    model_name,

    metrics,

    parameters,

    problem_type,

    dataset_shape
):

    create_experiment_directory()

    experiment_id = (
        generate_experiment_id()
    )

    sanitized_parameters = _sanitize_for_json(
        parameters
    )

    experiment_data = {

        "Experiment ID":
        experiment_id,

        "Timestamp":
        generate_timestamp(),

        "Model":
        model_name,

        "Problem Type":
        problem_type,

        "Dataset Rows":
        dataset_shape[0],

        "Dataset Columns":
        dataset_shape[1],

        "Metrics":
        metrics,

        "Parameters":
        sanitized_parameters
    }


    # -----------------------------------------------------
    # SAVE JSON
    # -----------------------------------------------------
    experiment_path = (
        f"experiments/"
        f"{experiment_id}.json"
    )

    with open(

        experiment_path,

        "w"
    ) as experiment_file:

        json.dump(

            experiment_data,

            experiment_file,

            indent=4
        )

    return experiment_data


# =========================================================
# LOAD ALL EXPERIMENTS
# =========================================================
def load_all_experiments():

    create_experiment_directory()

    experiments = []

    experiment_files = [

        file

        for file in os.listdir(
            "experiments"
        )

        if file.endswith(".json")
    ]


    # -----------------------------------------------------
    # LOAD FILES
    # -----------------------------------------------------
    for file in experiment_files:

        file_path = os.path.join(
            "experiments",
            file
        )

        try:

            with open(
                file_path,
                "r"
            ) as experiment_file:

                experiment_data = json.load(
                    experiment_file
                )

                experiments.append(
                    experiment_data
                )

        except Exception:

            continue


    return experiments


# =========================================================
# CONVERT TO DATAFRAME
# =========================================================
def experiments_to_dataframe(

    experiments
):

    rows = []

    for experiment in experiments:

        row = {

            "Experiment ID":
            experiment[
                "Experiment ID"
            ],

            "Timestamp":
            experiment[
                "Timestamp"
            ],

            "Model":
            experiment[
                "Model"
            ],

            "Problem Type":
            experiment[
                "Problem Type"
            ],

            "Parameter Count":
            len(
                experiment.get(
                    "Parameters",
                    {}
                )
            )
        }


        # -------------------------------------------------
        # FLATTEN METRICS
        # -------------------------------------------------
        metrics = experiment.get(
            "Metrics",
            {}
        )

        for metric_name, metric_value in (
            metrics.items()
        ):

            row[metric_name] = metric_value

        rows.append(row)


    dataframe = pd.DataFrame(
        rows
    )

    return dataframe


# =========================================================
# GET BEST EXPERIMENT
# =========================================================
def get_best_experiment(

    experiments_dataframe
):

    if experiments_dataframe.empty:

        return None


    # -----------------------------------------------------
    # POSSIBLE METRICS
    # -----------------------------------------------------
    possible_metrics = [

        "Accuracy",

        "R2 Score",

        "F1 Score"
    ]

    for metric in possible_metrics:

        if metric in experiments_dataframe.columns:

            best_row = (
                experiments_dataframe[
                    metric
                ].idxmax()
            )

            return experiments_dataframe.loc[
                best_row
            ]


    return None
