"""Feature-engineering helpers used by the Streamlit lab page."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PolynomialFeatures


TITLE_PATTERN = r",\s*([^\.]+)\."


class AutoMLFeatureEngineer(BaseEstimator, TransformerMixin):
    """Create stable raw-data features that can be reused at prediction time."""

    def fit(self, X: pd.DataFrame, y=None):
        dataframe = pd.DataFrame(X).copy()
        self.feature_names_in_ = list(dataframe.columns)
        self.generated_features_: list[str] = []
        self.common_titles_: set[str] = set()
        self.fare_bin_edges_: list[float] | None = None

        if "Name" in dataframe.columns:
            titles = self._extract_titles(dataframe["Name"])
            counts = titles.value_counts(dropna=True)
            self.common_titles_ = set(counts[counts >= max(5, int(len(dataframe) * 0.01))].index)

        if "Fare" in dataframe.columns:
            fare = pd.to_numeric(dataframe["Fare"], errors="coerce").dropna()
            if fare.nunique() >= 4:
                _, bins = pd.qcut(fare, q=4, duplicates="drop", retbins=True)
                unique_bins = sorted(set(float(value) for value in bins if np.isfinite(value)))
                if len(unique_bins) >= 2:
                    self.fare_bin_edges_ = [-np.inf, *unique_bins[1:-1], np.inf]

        transformed = self.transform(dataframe)
        self.generated_features_ = [
            column for column in transformed.columns if column not in self.feature_names_in_
        ]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        dataframe = pd.DataFrame(X).copy()

        for column in self.feature_names_in_:
            if column not in dataframe.columns:
                dataframe[column] = np.nan
        dataframe = dataframe[self.feature_names_in_]
        result = dataframe.copy()

        if {"SibSp", "Parch"}.issubset(result.columns):
            sibsp = pd.to_numeric(result["SibSp"], errors="coerce").fillna(0)
            parch = pd.to_numeric(result["Parch"], errors="coerce").fillna(0)
            family_size = sibsp + parch + 1
            result["FamilySize"] = family_size
            result["IsAlone"] = (family_size == 1).astype(int)

        if "Fare" in result.columns:
            fare = pd.to_numeric(result["Fare"], errors="coerce")
            denominator = result.get("FamilySize", pd.Series(1, index=result.index)).replace(0, np.nan)
            result["FarePerPerson"] = fare / denominator
            if self.fare_bin_edges_ is not None:
                result["FareBin"] = pd.cut(
                    fare,
                    bins=self.fare_bin_edges_,
                    labels=["Low", "Mid", "High", "Very High"][: len(self.fare_bin_edges_) - 1],
                    include_lowest=True,
                ).astype("string")

        if "Age" in result.columns:
            age = pd.to_numeric(result["Age"], errors="coerce")
            result["AgeGroup"] = pd.cut(
                age,
                bins=[-np.inf, 12, 18, 35, 60, np.inf],
                labels=["Child", "Teen", "Adult", "Middle", "Senior"],
            ).astype("string")

        if "Name" in result.columns:
            titles = self._extract_titles(result["Name"])
            if self.common_titles_:
                titles = titles.where(titles.isin(self.common_titles_), "Rare")
            result["Title"] = titles.fillna("Unknown").astype("string")

        if "Cabin" in result.columns:
            cabin = result["Cabin"].astype("string")
            result["CabinDeck"] = cabin.str.strip().str[0].fillna("Missing")
            result.loc[cabin.isna() | (cabin.str.strip() == ""), "CabinDeck"] = "Missing"

        if "Ticket" in result.columns:
            ticket = result["Ticket"].astype("string").str.upper().str.strip()
            prefix = ticket.str.replace(r"[\d\.\/\s]+", "", regex=True)
            result["TicketPrefix"] = prefix.where(prefix.str.len() > 0, "NUMERIC").fillna("Missing")

        if {"Pclass", "Sex"}.issubset(result.columns):
            result["Pclass_Sex"] = (
                result["Pclass"].astype("string").fillna("Missing")
                + "_"
                + result["Sex"].astype("string").fillna("Missing")
            )

        return result

    def get_feature_names_out(self, input_features=None):
        base_features = list(input_features) if input_features is not None else list(self.feature_names_in_)
        return np.asarray([*base_features, *self.generated_features_], dtype=object)

    @staticmethod
    def _extract_titles(series: pd.Series) -> pd.Series:
        return series.astype("string").str.extract(TITLE_PATTERN, expand=False).str.strip()


def get_numerical_columns(dataframe: pd.DataFrame) -> list[str]:
    return [
        column
        for column in dataframe.columns
        if is_numeric_dtype(dataframe[column]) and not pd.api.types.is_bool_dtype(dataframe[column])
    ]


def get_categorical_columns(dataframe: pd.DataFrame) -> list[str]:
    return [
        column
        for column in dataframe.columns
        if not is_numeric_dtype(dataframe[column]) and not is_datetime64_any_dtype(dataframe[column])
    ]


def detect_datetime_columns(dataframe: pd.DataFrame) -> list[str]:
    datetime_columns: list[str] = []
    for column in dataframe.columns:
        series = dataframe[column]
        if is_datetime64_any_dtype(series):
            datetime_columns.append(column)
            continue
        if series.dtype == "object":
            parsed = pd.to_datetime(series, errors="coerce")
            if parsed.notna().mean() >= 0.8 and parsed.nunique(dropna=True) > 1:
                datetime_columns.append(column)
    return datetime_columns


def create_interaction_feature(
    dataframe: pd.DataFrame,
    column_1: str,
    column_2: str,
    operation: str,
) -> pd.DataFrame:
    result = dataframe.copy()
    safe_left = pd.to_numeric(result[column_1], errors="coerce")
    safe_right = pd.to_numeric(result[column_2], errors="coerce")

    if operation == "add":
        values = safe_left + safe_right
        suffix = "plus"
    elif operation == "subtract":
        values = safe_left - safe_right
        suffix = "minus"
    elif operation == "multiply":
        values = safe_left * safe_right
        suffix = "times"
    elif operation == "divide":
        values = safe_left / safe_right.replace(0, np.nan)
        suffix = "div"
    else:
        raise ValueError(f"Unsupported interaction operation: {operation}")

    result[f"{column_1}_{suffix}_{column_2}"] = values
    return result


def create_polynomial_features(
    dataframe: pd.DataFrame,
    selected_columns: list[str],
    degree: int,
) -> pd.DataFrame:
    if not selected_columns:
        raise ValueError("Select at least one numerical column.")

    result = dataframe.copy()
    transformer = PolynomialFeatures(
        degree=degree,
        include_bias=False,
    )
    numeric_frame = result[selected_columns].apply(pd.to_numeric, errors="coerce").fillna(0)
    transformed = transformer.fit_transform(numeric_frame)
    feature_names = transformer.get_feature_names_out(selected_columns)

    for index, feature_name in enumerate(feature_names):
        if feature_name not in selected_columns:
            normalized_name = feature_name.replace(" ", "_").replace("^", "_pow_")
            result[normalized_name] = transformed[:, index]

    return result


def apply_log_transformation(
    dataframe: pd.DataFrame,
    selected_column: str,
) -> pd.DataFrame:
    result = dataframe.copy()
    numeric_values = pd.to_numeric(result[selected_column], errors="coerce")
    if numeric_values.dropna().empty:
        raise ValueError("Selected column does not contain numeric values.")
    shift = 1 - numeric_values.min() if numeric_values.min() <= 0 else 0
    result[f"{selected_column}_log"] = np.log1p(numeric_values + shift)
    return result


def apply_feature_binning(
    dataframe: pd.DataFrame,
    selected_column: str,
    bins: int,
) -> pd.DataFrame:
    result = dataframe.copy()
    numeric_values = pd.to_numeric(result[selected_column], errors="coerce")
    if numeric_values.nunique(dropna=True) < 2:
        raise ValueError("Selected column needs at least two distinct numeric values.")
    result[f"{selected_column}_bin"] = pd.cut(
        numeric_values,
        bins=min(bins, numeric_values.nunique(dropna=True)),
        duplicates="drop",
    ).astype("string")
    return result


def apply_automl_feature_engineering(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Apply the same practical AutoML features used by the training pipeline."""

    transformer = AutoMLFeatureEngineer()
    return transformer.fit_transform(dataframe)


def _add_family_size_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    if not {"SibSp", "Parch"}.issubset(dataframe.columns):
        return dataframe.copy()

    result = dataframe.copy()
    sibsp = pd.to_numeric(result["SibSp"], errors="coerce").fillna(0)
    parch = pd.to_numeric(result["Parch"], errors="coerce").fillna(0)
    family_size = sibsp + parch + 1
    result["FamilySize"] = family_size
    result["IsAlone"] = (family_size == 1).astype(int)
    return result


def _add_fare_per_person_feature(dataframe: pd.DataFrame) -> pd.DataFrame:
    if "Fare" not in dataframe.columns:
        return dataframe.copy()

    result = dataframe.copy()
    fare = pd.to_numeric(result["Fare"], errors="coerce")
    denominator = result.get("FamilySize", pd.Series(1, index=result.index)).replace(0, np.nan)
    result["FarePerPerson"] = fare / denominator
    return result


def _add_age_group_feature(dataframe: pd.DataFrame) -> pd.DataFrame:
    if "Age" not in dataframe.columns:
        return dataframe.copy()

    result = dataframe.copy()
    age = pd.to_numeric(result["Age"], errors="coerce")
    result["AgeGroup"] = pd.cut(
        age,
        bins=[-np.inf, 12, 18, 35, 60, np.inf],
        labels=["Child", "Teen", "Adult", "Middle", "Senior"],
    ).astype("string")
    return result


def _add_title_feature(dataframe: pd.DataFrame) -> pd.DataFrame:
    if "Name" not in dataframe.columns:
        return dataframe.copy()

    result = dataframe.copy()
    titles = result["Name"].astype("string").str.extract(TITLE_PATTERN, expand=False).str.strip()
    counts = titles.value_counts(dropna=True)
    common_titles = set(counts[counts >= max(5, int(len(result) * 0.01))].index)
    if common_titles:
        titles = titles.where(titles.isin(common_titles), "Rare")
    result["Title"] = titles.fillna("Unknown").astype("string")
    return result


def _add_cabin_deck_feature(dataframe: pd.DataFrame) -> pd.DataFrame:
    if "Cabin" not in dataframe.columns:
        return dataframe.copy()

    result = dataframe.copy()
    cabin = result["Cabin"].astype("string")
    result["CabinDeck"] = cabin.str.strip().str[0].fillna("Missing")
    result.loc[cabin.isna() | (cabin.str.strip() == ""), "CabinDeck"] = "Missing"
    return result


def _add_ticket_prefix_feature(dataframe: pd.DataFrame) -> pd.DataFrame:
    if "Ticket" not in dataframe.columns:
        return dataframe.copy()

    result = dataframe.copy()
    ticket = result["Ticket"].astype("string").str.upper().str.strip()
    prefix = ticket.str.replace(r"[\d\.\/\s]+", "", regex=True)
    result["TicketPrefix"] = prefix.where(prefix.str.len() > 0, "NUMERIC").fillna("Missing")
    return result


def _add_pclass_sex_feature(dataframe: pd.DataFrame) -> pd.DataFrame:
    if not {"Pclass", "Sex"}.issubset(dataframe.columns):
        return dataframe.copy()

    result = dataframe.copy()
    result["Pclass_Sex"] = (
        result["Pclass"].astype("string").fillna("Missing")
        + "_"
        + result["Sex"].astype("string").fillna("Missing")
    )
    return result


def get_ai_feature_recommendations(dataframe: pd.DataFrame) -> list[dict[str, str]]:
    """Return guided, manually applicable feature recommendations."""

    recommendations: list[dict[str, str]] = []

    if {"SibSp", "Parch"}.issubset(dataframe.columns):
        recommendations.append(
            {
                "id": "family_size",
                "label": "Family Size",
                "description": "Combine SibSp and Parch into FamilySize and IsAlone.",
                "expected_benefit": "Helps the model capture passenger group context.",
            }
        )

    if "Fare" in dataframe.columns:
        recommendations.append(
            {
                "id": "fare_binning",
                "label": "Fare Binning",
                "description": "Create grouped fare ranges from the Fare column.",
                "expected_benefit": "Can reduce skew and make fare levels easier to learn.",
            }
        )
        recommendations.append(
            {
                "id": "fare_per_person",
                "label": "Fare Per Person",
                "description": "Divide fare by family size or treat it as an individual fare estimate.",
                "expected_benefit": "Adds context when tickets are shared across passengers.",
            }
        )

    if "Age" in dataframe.columns:
        recommendations.append(
            {
                "id": "age_groups",
                "label": "Age Groups",
                "description": "Bucket Age into interpretable groups such as Child, Adult, and Senior.",
                "expected_benefit": "Makes non-linear age effects easier for simpler models to learn.",
            }
        )

    if "Name" in dataframe.columns:
        recommendations.append(
            {
                "id": "title_extraction",
                "label": "Title Extraction",
                "description": "Extract a cleaned social title from the Name column.",
                "expected_benefit": "Turns noisy text into a smaller, more informative categorical feature.",
            }
        )

    if "Cabin" in dataframe.columns:
        recommendations.append(
            {
                "id": "cabin_deck",
                "label": "Cabin Deck",
                "description": "Capture the first cabin letter as a deck-level feature.",
                "expected_benefit": "Keeps useful location signal while simplifying sparse cabin values.",
            }
        )

    if "Ticket" in dataframe.columns:
        recommendations.append(
            {
                "id": "ticket_prefix",
                "label": "Ticket Prefix",
                "description": "Extract the non-numeric ticket prefix from Ticket.",
                "expected_benefit": "Can preserve booking-pattern signal without using the raw identifier.",
            }
        )

    if {"Pclass", "Sex"}.issubset(dataframe.columns):
        recommendations.append(
            {
                "id": "pclass_sex",
                "label": "Interaction Feature",
                "description": "Create a combined Pclass_Sex feature.",
                "expected_benefit": "Useful when class and sex jointly affect the target.",
            }
        )

    return recommendations


def apply_ai_recommended_features(
    dataframe: pd.DataFrame,
    selected_recommendation_ids: list[str],
) -> pd.DataFrame:
    """Apply only the feature recommendations selected by the user."""

    result = dataframe.copy()

    for recommendation_id in selected_recommendation_ids:
        if recommendation_id == "family_size":
            result = _add_family_size_features(result)
        elif recommendation_id == "fare_binning":
            result = apply_feature_binning(result, "Fare", bins=4)
        elif recommendation_id == "fare_per_person":
            result = _add_family_size_features(result)
            result = _add_fare_per_person_feature(result)
        elif recommendation_id == "age_groups":
            result = _add_age_group_feature(result)
        elif recommendation_id == "title_extraction":
            result = _add_title_feature(result)
        elif recommendation_id == "cabin_deck":
            result = _add_cabin_deck_feature(result)
        elif recommendation_id == "ticket_prefix":
            result = _add_ticket_prefix_feature(result)
        elif recommendation_id == "pclass_sex":
            result = _add_pclass_sex_feature(result)

    return result


def extract_datetime_features(
    dataframe: pd.DataFrame,
    selected_datetime_column: str,
) -> pd.DataFrame:
    result = dataframe.copy()
    parsed = pd.to_datetime(result[selected_datetime_column], errors="coerce")
    if parsed.notna().sum() == 0:
        raise ValueError("Selected column could not be parsed as datetime.")

    result[f"{selected_datetime_column}_year"] = parsed.dt.year
    result[f"{selected_datetime_column}_month"] = parsed.dt.month
    result[f"{selected_datetime_column}_day"] = parsed.dt.day
    result[f"{selected_datetime_column}_dayofweek"] = parsed.dt.dayofweek
    return result


def generate_feature_engineering_suggestions(dataframe: pd.DataFrame) -> list[str]:
    suggestions: list[str] = []
    numerical_columns = get_numerical_columns(dataframe)
    datetime_columns = detect_datetime_columns(dataframe)

    for recommendation in get_ai_feature_recommendations(dataframe):
        suggestions.append(
            f"{recommendation['label']}: {recommendation['description']} Expected benefit: {recommendation['expected_benefit']}"
        )

    if len(numerical_columns) >= 2:
        suggestions.append("Consider interaction features for related numerical predictors.")

    for column in numerical_columns:
        skewness = dataframe[column].dropna().skew()
        if pd.notna(skewness) and abs(skewness) > 1:
            suggestions.append(f"Column '{column}' is skewed; a log transform may help.")

    if datetime_columns:
        suggestions.append("Datetime columns detected; extracting calendar features can improve signal.")

    return suggestions


def generate_feature_engineering_summary(dataframe: pd.DataFrame) -> dict[str, int]:
    numerical_columns = get_numerical_columns(dataframe)
    skewed_features = sum(
        abs(dataframe[column].dropna().skew()) > 1
        for column in numerical_columns
        if dataframe[column].dropna().shape[0] > 2
    )
    return {
        "total_rows": dataframe.shape[0],
        "total_columns": dataframe.shape[1],
        "numerical_columns": len(numerical_columns),
        "categorical_columns": len(get_categorical_columns(dataframe)),
        "datetime_columns": len(detect_datetime_columns(dataframe)),
        "skewed_features": int(skewed_features),
    }
