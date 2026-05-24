# =========================================================
# FILE: src/utils.py
# =========================================================

import os
import pandas as pd
import numpy as np

from datetime import datetime


# =========================================================
# GENERATE TIMESTAMP
# =========================================================
def generate_timestamp():
    """
    Generates formatted timestamp.
    """

    return datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )


# =========================================================
# CREATE DIRECTORY
# =========================================================
def create_directory(

    directory_path
):
    """
    Creates directory safely.
    """

    os.makedirs(
        directory_path,
        exist_ok=True
    )


# =========================================================
# FORMAT LARGE NUMBER
# =========================================================
def format_large_number(

    value
):
    """
    Formats large numbers professionally.
    """

    if value >= 1_000_000:

        return f"{value/1_000_000:.2f}M"

    elif value >= 1_000:

        return f"{value/1_000:.2f}K"

    return str(value)


# =========================================================
# DETECT NUMERICAL COLUMNS
# =========================================================
def get_numerical_columns(

    dataframe
):
    """
    Returns numerical columns.
    """

    numerical_columns = (
        dataframe.select_dtypes(
            include=np.number
        ).columns.tolist()
    )

    return numerical_columns


# =========================================================
# DETECT CATEGORICAL COLUMNS
# =========================================================
def get_categorical_columns(

    dataframe
):
    """
    Returns categorical columns.
    """

    categorical_columns = (
        dataframe.select_dtypes(
            exclude=np.number
        ).columns.tolist()
    )

    return categorical_columns


# =========================================================
# GET MEMORY USAGE
# =========================================================
def calculate_memory_usage(

    dataframe
):
    """
    Returns dataframe memory usage in MB.
    """

    memory_usage = (
        dataframe.memory_usage(
            deep=True
        ).sum()
        /
        (1024 ** 2)
    )

    return round(
        memory_usage,
        2
    )


# =========================================================
# MISSING VALUE SUMMARY
# =========================================================
def generate_missing_value_summary(

    dataframe
):
    """
    Generates missing value summary.
    """

    missing_dataframe = pd.DataFrame({

        "Column":
        dataframe.columns,

        "Missing Values":
        dataframe.isnull().sum().values,

        "Missing Percentage":
        (
            dataframe.isnull().mean()
            * 100
        ).values
    })

    missing_dataframe = (
        missing_dataframe.sort_values(

            by="Missing Values",

            ascending=False
        )
    )

    return missing_dataframe


# =========================================================
# SAFE DATAFRAME COPY
# =========================================================
def safe_copy_dataframe(

    dataframe
):
    """
    Safely copies dataframe.
    """

    return dataframe.copy()


# =========================================================
# CLEAN COLUMN NAMES
# =========================================================
def clean_column_names(

    dataframe
):
    """
    Cleans dataframe column names.
    """

    dataframe.columns = [

        column.strip()
        .lower()
        .replace(" ", "_")

        for column in dataframe.columns
    ]

    return dataframe


# =========================================================
# CHECK FILE EXISTS
# =========================================================
def check_file_exists(

    file_path
):
    """
    Checks whether file exists.
    """

    return os.path.exists(
        file_path
    )


# =========================================================
# GET DATAFRAME SHAPE SUMMARY
# =========================================================
def generate_shape_summary(

    dataframe
):
    """
    Generates dataframe shape summary.
    """

    summary = {

        "Rows":
        dataframe.shape[0],

        "Columns":
        dataframe.shape[1]
    }

    return summary


# =========================================================
# DETECT DUPLICATES
# =========================================================
def count_duplicate_rows(

    dataframe
):
    """
    Counts duplicate rows.
    """

    return dataframe.duplicated().sum()


# =========================================================
# FORMAT METRIC VALUE
# =========================================================
def format_metric_value(

    value
):
    """
    Formats metric values.
    """

    if isinstance(
        value,
        float
    ):

        return round(
            value,
            4
        )

    return value