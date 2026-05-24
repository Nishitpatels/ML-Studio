"""Gemini-backed AI helpers with graceful configuration failure."""

from __future__ import annotations

import os
import warnings
from functools import lru_cache

from dotenv import load_dotenv

from src.constants import DEFAULT_GEMINI_MODEL

try:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        import google.generativeai as genai
except Exception:  # pragma: no cover - optional dependency in some environments
    genai = None


load_dotenv()

FALLBACK_GEMINI_MODELS = (
    DEFAULT_GEMINI_MODEL,
    "gemini-flash-latest",
)


class AIServiceUnavailableError(RuntimeError):
    """Raised when the AI assistant cannot be used safely."""


def get_ai_configuration_error() -> str | None:
    if genai is None:
        return "google-generativeai is not installed."
    if not os.getenv("GEMINI_API_KEY"):
        return "GEMINI_API_KEY is not configured in the .env file."
    return None


def is_ai_configured() -> bool:
    return get_ai_configuration_error() is None


def get_ai_model_label() -> str:
    configured_model = os.getenv("GEMINI_MODEL")
    return configured_model or DEFAULT_GEMINI_MODEL


def _get_model_candidates() -> list[str]:
    candidates: list[str] = []
    configured_model = os.getenv("GEMINI_MODEL")
    for model_name in (configured_model, *FALLBACK_GEMINI_MODELS):
        if model_name and model_name not in candidates:
            candidates.append(model_name)
    return candidates


@lru_cache(maxsize=8)
def _build_model(api_key: str, model_name: str):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name)


def _is_model_lookup_error(error: Exception) -> bool:
    message = str(error).lower()
    return (
        "not found" in message
        or "404" in message
        or "unsupported for generatecontent" in message
        or "model is not found" in message
    )


def _normalize_ai_error(error: Exception) -> str:
    message = str(error)
    lowered_message = message.lower()

    if _is_model_lookup_error(error):
        return "The configured Gemini model is unavailable. ML Studio tried supported fallback models but could not reach a valid text model."
    if "api key" in lowered_message or "permission" in lowered_message or "401" in lowered_message or "403" in lowered_message:
        return "Gemini authentication failed. Check GEMINI_API_KEY in the .env file."
    if "quota" in lowered_message or "429" in lowered_message or "rate limit" in lowered_message:
        return "Gemini rate limits or quota were reached. Please wait and try again."
    if "deadline" in lowered_message or "timeout" in lowered_message or "503" in lowered_message or "unavailable" in lowered_message:
        return "Gemini is temporarily unavailable. Please try again shortly."
    return f"Gemini request failed: {message}"


def _extract_text(response) -> str:
    try:
        text = getattr(response, "text", None)
        if text and text.strip():
            return text.strip()
    except Exception:
        pass

    candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) if content is not None else None
        text_parts = [
            getattr(part, "text", "")
            for part in (parts or [])
            if getattr(part, "text", "")
        ]
        if text_parts:
            return "\n".join(text_parts).strip()

    raise AIServiceUnavailableError(
        "Gemini returned an empty response. Try again or adjust the prompt."
    )


def _generate(prompt: str) -> str:
    configuration_error = get_ai_configuration_error()
    if configuration_error:
        raise AIServiceUnavailableError(configuration_error)

    api_key = os.getenv("GEMINI_API_KEY")
    attempted_models: list[str] = []
    last_error: Exception | None = None

    for model_name in _get_model_candidates():
        attempted_models.append(model_name)
        try:
            response = _build_model(api_key, model_name).generate_content(prompt)
            return _extract_text(response)
        except AIServiceUnavailableError as error:
            last_error = error
            break
        except Exception as error:
            last_error = error
            if _is_model_lookup_error(error):
                continue
            raise AIServiceUnavailableError(_normalize_ai_error(error)) from error

    if last_error is not None:
        raise AIServiceUnavailableError(
            f"{_normalize_ai_error(last_error)} Models tried: {', '.join(attempted_models)}."
        ) from last_error

    raise AIServiceUnavailableError("Gemini did not return a usable response.")


def generate_dataset_summary(dataframe):
    return _generate(
        f"""
        You are an expert data scientist.
        Rows: {dataframe.shape[0]}
        Columns: {dataframe.shape[1]}
        Column Names: {list(dataframe.columns)}
        Missing Values: {dataframe.isnull().sum().to_dict()}
        Data Types: {dataframe.dtypes.astype(str).to_dict()}
        Give: dataset overview, likely use cases, data-quality issues, preprocessing suggestions, and feature ideas.
        Keep the response professional and concise.
        """
    )


def generate_model_recommendations(dataframe, target_column):
    return _generate(
        f"""
        You are an ML expert.
        Dataset Columns: {list(dataframe.columns)}
        Target Column: {target_column}
        Data Types: {dataframe.dtypes.astype(str).to_dict()}
        Recommend suitable models, why they fit, preprocessing, feature engineering, and likely challenges.
        Keep the response concise and practical.
        """
    )


def generate_eda_insights(dataframe):
    numerical_summary = dataframe.describe(include="all").transpose().fillna("").to_string()
    return _generate(
        f"""
        Analyze this dataset.
        Shape: {dataframe.shape}
        Missing Values: {dataframe.isnull().sum().to_dict()}
        Summary:
        {numerical_summary}
        Give EDA insights, outlier concerns, skewness concerns, correlation ideas, and transformation suggestions.
        """
    )


def ask_ai_question(question, dataframe):
    return _generate(
        f"""
        You are an expert AI data scientist.
        Columns: {list(dataframe.columns)}
        Shape: {dataframe.shape}
        User Question: {question}
        Give a helpful answer.
        """
    )


def chat_with_dataset(dataframe, user_question, chat_history=None):
    chat_history = chat_history or []
    history_lines = []
    for message in chat_history[-8:]:
        if isinstance(message, dict):
            role = message.get("role", "user")
            content = message.get("content", "")
        else:
            role = "user"
            content = str(message)
        if content:
            history_lines.append(f"{role}: {content}")
    history = "\n".join(history_lines)
    return _generate(
        f"""
        You are an expert AI data scientist.
        Dataset Shape: {dataframe.shape}
        Columns: {list(dataframe.columns)}
        Missing Values: {dataframe.isnull().sum().to_dict()}
        Data Types: {dataframe.dtypes.astype(str).to_dict()}
        Sample Data:
        {dataframe.head(5).to_string()}
        Chat History:
        {history}
        User Question:
        {user_question}
        Give practical, concise ML/data-science guidance.
        """
    )


def generate_ai_insights(dataframe):
    return _generate(
        f"""
        You are a principal data scientist.
        Dataset Shape: {dataframe.shape}
        Columns: {list(dataframe.columns)}
        Missing Values: {dataframe.isnull().sum().to_dict()}
        Data Types: {dataframe.dtypes.astype(str).to_dict()}
        Generate advanced insights: data quality, feature hypotheses, business questions, modeling risks, and next actions.
        """
    )
