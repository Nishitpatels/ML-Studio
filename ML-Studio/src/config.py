# =========================================================
# FILE: src/config.py
# =========================================================

import os

from dotenv import load_dotenv

from src.constants import (


    APP_NAME,

    APP_VERSION,

    RANDOM_STATE,

    DEFAULT_TEST_SIZE
)


# ---------------------------------------------------------
# LOAD ENVIRONMENT VARIABLES
# ---------------------------------------------------------
load_dotenv()


# =========================================================
# APPLICATION CONFIG
# =========================================================
CONFIG = {

    # -----------------------------------------------------
    # APP INFO
    # -----------------------------------------------------
    "APP_NAME":
    APP_NAME,

    "APP_VERSION":
    APP_VERSION,


    # -----------------------------------------------------
    # RANDOM SETTINGS
    # -----------------------------------------------------
    "RANDOM_STATE":
    RANDOM_STATE,

    "TEST_SIZE":
    DEFAULT_TEST_SIZE,


    # -----------------------------------------------------
    # GEMINI API
    # -----------------------------------------------------
    "GEMINI_API_KEY":
    os.getenv(
        "GEMINI_API_KEY"
    ),


    # -----------------------------------------------------
    # DIRECTORIES
    # -----------------------------------------------------
    "REPORTS_DIR":
    "reports",

    "MODELS_DIR":
    "models",

    "EXPERIMENTS_DIR":
    "experiments"
}


# =========================================================
# VALIDATE CONFIG
# =========================================================
def validate_config():
    """
    Validates required configuration.
    """

    validation_results = {

        "Gemini API Configured":
        CONFIG["GEMINI_API_KEY"] is not None
    }

    return validation_results


# =========================================================
# GET CONFIG VALUE
# =========================================================
def get_config(

    key,

    default=None
):
    """
    Safely fetch configuration values.
    """

    return CONFIG.get(
        key,
        default
    )