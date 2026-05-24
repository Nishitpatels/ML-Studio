# =========================================================
# FILE: pages/11_Predictions.py
# =========================================================


import pandas as pd
import streamlit as st

from src.prediction import (
    build_prediction_explanation,
    generate_prediction_summary,
    get_expected_features,
    get_prediction_probabilities,
    make_batch_predictions_with_probabilities,
    make_single_prediction,
    prepare_prediction_input,
)
from src.session_manager import get_missing_or_empty_keys, initialize_session_state
from src.ui_helpers import load_ml_studio_css, render_engineered_data_section, render_page_header


st.set_page_config(
    page_title="Predictions | ML Studio",
    page_icon=":material/online_prediction:",
    layout="wide"
)

load_ml_studio_css()
render_page_header(
    "Live Predictions",
    "Generate single or batch predictions using the currently active trained model.",
    "Prediction",
)
initialize_session_state(st.session_state)

required_keys = [
    "best_model",
    "preprocessing_results",
]

missing_keys = get_missing_or_empty_keys(st.session_state, required_keys)
if missing_keys:
    st.warning(f"Missing required workflow artifacts: {', '.join(missing_keys)}")
    st.stop()

best_model_results = st.session_state.best_model
preprocessing_results = st.session_state.preprocessing_results
target_column = st.session_state.get("target_column")

pipeline = best_model_results["pipeline"]
X_train = preprocessing_results["X_train"]
raw_reference = preprocessing_results.get(
    "cleaned_dataframe",
    pd.concat(
        [
            preprocessing_results.get("X_train_raw", X_train),
            preprocessing_results.get("X_test_raw", pd.DataFrame()),
        ]
    ),
)
if target_column in raw_reference.columns:
    raw_reference = raw_reference.drop(columns=[target_column])

expected_features = get_expected_features(pipeline) or list(X_train.columns)
user_input_columns = [
    column
    for column in preprocessing_results.get("prediction_input_columns", expected_features)
    if column in expected_features
]
if not user_input_columns:
    user_input_columns = expected_features
hidden_columns = [column for column in expected_features if column not in user_input_columns]
model_classes = getattr(pipeline.named_steps.get("model"), "classes_", None)

with st.sidebar:
    st.markdown("## Prediction Controls")
    prediction_mode = st.radio(
        "Prediction Mode",
        options=[
            "Single Prediction",
            "Batch Prediction"
        ]
    )

if prediction_mode == "Single Prediction":
    st.markdown("---")
    st.subheader("Single Prediction")

    input_data = {}
    if hidden_columns:
        st.info(
            "ML Studio will handle ignored low-value fields internally: "
            + ", ".join(hidden_columns)
        )

    for column in user_input_columns:
        reference_series = (
            raw_reference[column]
            if column in raw_reference.columns
            else pd.Series(dtype="object")
        )
        column_dtype = str(reference_series.dtype)

        if "int" in column_dtype or "float" in column_dtype:
            numeric_series = pd.to_numeric(reference_series, errors="coerce")
            default_value = numeric_series.dropna().median()
            if pd.isna(default_value):
                default_value = 0.0

            input_data[column] = st.number_input(
                label=column,
                value=float(default_value)
            )
        else:
            unique_values = (
                reference_series
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            if unique_values and len(unique_values) <= 50:
                input_data[column] = st.selectbox(
                    label=column,
                    options=sorted(unique_values)
                )
            else:
                input_data[column] = st.text_input(
                    label=column,
                    value=str(unique_values[0]) if unique_values else ""
                )

    predict_button = st.button("Predict")

    if predict_button:
        try:
            input_dataframe, _, _ = prepare_prediction_input(pd.DataFrame([input_data]), expected_features)
            prediction = make_single_prediction(pipeline, input_dataframe)
            probabilities = get_prediction_probabilities(pipeline, input_dataframe)
            explanation = build_prediction_explanation(
                prediction,
                target_column=target_column,
                probabilities=probabilities,
                classes=model_classes,
            )

            st.markdown("---")
            st.subheader("Prediction Result")
            headline_col1, headline_col2 = st.columns([1.4, 1.0])
            headline_col1.success(explanation["interpretation"])
            if explanation["confidence"] is not None:
                headline_col2.metric("Confidence", f"{explanation['confidence']:.2%}")
            else:
                headline_col2.metric("Confidence", "N/A")

            detail_col1, detail_col2 = st.columns(2)
            detail_col1.metric("Raw Prediction", explanation["raw_prediction"])
            detail_col2.metric("Target", target_column or "Outcome")
            st.info(explanation["explanation"])

            if probabilities is not None:
                probability_dataframe = pd.DataFrame(
                    {
                        "Class": list(model_classes) if model_classes is not None else list(range(len(probabilities))),
                        "Probability": probabilities,
                    }
                ).sort_values("Probability", ascending=False)
                st.dataframe(probability_dataframe, width="stretch")

        except Exception as error:
            st.error(f"Prediction failed: {error}")

elif prediction_mode == "Batch Prediction":
    st.markdown("---")
    st.subheader("Batch Prediction")

    uploaded_prediction_file = st.file_uploader(
        "Upload Prediction CSV",
        type=["csv"]
    )

    if uploaded_prediction_file is not None:
        try:
            prediction_dataframe = pd.read_csv(uploaded_prediction_file)

            st.markdown("---")
            st.subheader("Uploaded Prediction Data")
            st.dataframe(prediction_dataframe.head(), width="stretch")

            prepared_dataframe, missing_columns, extra_columns = prepare_prediction_input(
                prediction_dataframe,
                expected_features,
            )
            if missing_columns:
                visible_missing = [column for column in missing_columns if column in user_input_columns]
                hidden_missing = [column for column in missing_columns if column not in user_input_columns]
                if visible_missing:
                    st.warning(f"Missing columns were added as nulls and will be imputed: {visible_missing}")
                if hidden_missing:
                    st.info(f"Ignored low-value columns were not required for prediction: {hidden_missing}")
            if extra_columns:
                st.info(f"Extra columns were ignored: {extra_columns}")

            batch_predict_button = st.button("Generate Predictions")

            if batch_predict_button:
                with st.spinner("Generating predictions..."):
                    prediction_results = make_batch_predictions_with_probabilities(
                        pipeline,
                        prepared_dataframe
                    )
                    prediction_results["Prediction Label"] = prediction_results["Prediction"].apply(
                        lambda value: build_prediction_explanation(
                            value,
                            target_column=target_column,
                        )["interpretation"]
                    )

                    st.session_state["prediction_results"] = prediction_results

                    st.success("Predictions generated successfully.")

                    st.markdown("---")
                    st.subheader("Prediction Results")
                    st.dataframe(prediction_results, width="stretch")

                    st.markdown("---")
                    st.subheader("Prediction Summary")
                    summary = generate_prediction_summary(prediction_results)

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Total Predictions", summary["Total Predictions"])
                    with col2:
                        st.metric("Columns", summary["Prediction Columns"])

                    st.markdown("---")
                    csv_data = prediction_results.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="Download Predictions CSV",
                        data=csv_data,
                        file_name="prediction_results.csv",
                        mime="text/csv"
                    )

        except Exception as error:
            st.error(f"Batch prediction failed: {error}")

render_engineered_data_section(session_state=st.session_state)
st.markdown("---")
st.success("Prediction system ready.")
