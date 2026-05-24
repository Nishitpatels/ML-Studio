# =========================================================
# FILE: src/settings_manager.py
# =========================================================

import os
import shutil


# =========================================================
# CLEAR STREAMLIT SESSION
# =========================================================
def clear_session_state(

    session_state
):

    keys = list(
        session_state.keys()
    )

    for key in keys:

        del session_state[key]


# =========================================================
# DELETE DIRECTORY CONTENTS
# =========================================================
def clear_directory(

    directory_path
):

    if not os.path.exists(
        directory_path
    ):

        return False


    for item in os.listdir(
        directory_path
    ):

        item_path = os.path.join(

            directory_path,

            item
        )

        try:

            if os.path.isfile(
                item_path
            ):

                os.remove(
                    item_path
                )

            elif os.path.isdir(
                item_path
            ):

                shutil.rmtree(
                    item_path
                )

        except Exception:

            continue


    return True


# =========================================================
# GET DIRECTORY FILE COUNT
# =========================================================
def get_directory_file_count(

    directory_path
):

    if not os.path.exists(
        directory_path
    ):

        return 0

    return len(
        os.listdir(directory_path)
    )


# =========================================================
# GET SYSTEM STORAGE SUMMARY
# =========================================================
def generate_storage_summary():

    directories = {

        "Reports":
        "reports",

        "Experiments":
        "experiments",

        "Exported Models":
        "models/exported",

        "Metadata":
        "models/metadata"
    }

    summary = {}

    for name, path in directories.items():

        summary[name] = (
            get_directory_file_count(
                path
            )
        )

    return summary


# =========================================================
# INITIALIZE REQUIRED DIRECTORIES
# =========================================================
def initialize_directories():

    required_directories = [

        "reports",

        "experiments",

        "models",

        "models/exported",

        "models/metadata"
    ]

    for directory in required_directories:

        os.makedirs(
            directory,
            exist_ok=True
        )