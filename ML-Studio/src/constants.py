# =========================================================
# FILE: src/constants.py
# =========================================================

"""
Global constants used across ML Studio
"""


# =========================================================
# APP INFORMATION
# =========================================================
APP_NAME = "ML Studio"

APP_VERSION = "1.0.0"

APP_DESCRIPTION = (
    "Enterprise AutoML Platform"
)


# =========================================================
# RANDOM STATE
# =========================================================
RANDOM_STATE = 42


# =========================================================
# DATASET SETTINGS
# =========================================================
DEFAULT_TEST_SIZE = 0.2

DEFAULT_CV_FOLDS = 5

DEFAULT_TOP_FEATURES = 20


# =========================================================
# FILE SETTINGS
# =========================================================
SUPPORTED_FILE_TYPES = [

    "csv"
]


# =========================================================
# MODEL TYPES
# =========================================================
CLASSIFICATION = "classification"

REGRESSION = "regression"


# =========================================================
# CLASSIFICATION MODELS
# =========================================================
CLASSIFICATION_MODEL_NAMES = [

    "Logistic Regression",

    "Decision Tree",

    "Random Forest",

    "Gradient Boosting",

    "SVM"
]


# =========================================================
# REGRESSION MODELS
# =========================================================
REGRESSION_MODEL_NAMES = [

    "Linear Regression",

    "Decision Tree Regressor",

    "Random Forest Regressor",

    "Gradient Boosting Regressor"
]


# =========================================================
# DRIFT THRESHOLDS
# =========================================================
PSI_NO_DRIFT_THRESHOLD = 0.1

PSI_MODERATE_DRIFT_THRESHOLD = 0.25


# =========================================================
# DIRECTORY PATHS
# =========================================================
REPORTS_DIRECTORY = "reports"

EXPERIMENTS_DIRECTORY = "experiments"

MODELS_DIRECTORY = "models"

EXPORTED_MODELS_DIRECTORY = (
    "models/exported"
)

METADATA_DIRECTORY = (
    "models/metadata"
)


# =========================================================
# SESSION KEYS
# =========================================================
SESSION_DATASET = "dataset"

SESSION_TARGET_COLUMN = (
    "target_column"
)

SESSION_PREPROCESSING_RESULTS = (
    "preprocessing_results"
)

SESSION_TRAINING_RESULTS = (
    "training_results"
)

SESSION_BEST_MODEL = (
    "best_model"
)


# =========================================================
# AI SETTINGS
# =========================================================
DEFAULT_GEMINI_MODEL = (
    "gemini-2.5-flash"
)


# =========================================================
# CHART SETTINGS
# =========================================================
DEFAULT_CHART_HEIGHT = 600
