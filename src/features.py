"""Shared feature engineering and model-specific preprocessing builders."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

from . import config


def _as_frame(X: pd.DataFrame | np.ndarray, columns: Iterable[str] | None = None) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X
    return pd.DataFrame(X, columns=list(columns) if columns is not None else None)


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Create deterministic, leak-safe engineered features from raw rows.

    The expected-kilometres baseline is learned from the training data during
    ``fit`` and stored for later use.
    """

    def __init__(self, add_interactions: bool = False) -> None:
        self.add_interactions = add_interactions

    def fit(self, X: pd.DataFrame, y: object = None) -> "FeatureEngineer":
        X = _as_frame(X)
        age = config.REF_YEAR - X["Manufacturing_Year"].astype(int)
        medians = pd.DataFrame(
            {
                "Age": age.to_numpy(),
                "Kilometres_Driven": X["Kilometres_Driven"].to_numpy(dtype=float),
            }
        ).groupby("Age")["Kilometres_Driven"].median()
        self.expected_km_by_age_ = medians.to_dict()
        self.global_median_km_ = float(X["Kilometres_Driven"].median())
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = _as_frame(X)
        out = X.copy()
        age = (config.REF_YEAR - out["Manufacturing_Year"].astype(int)).astype(float)
        ownership_age = (config.REF_YEAR - out["Year_Purchased"].astype(int)).astype(float)
        km = out["Kilometres_Driven"].astype(float)
        expected = age.map(self.expected_km_by_age_).astype(float)
        expected = expected.fillna(self.global_median_km_).clip(lower=1.0)

        out["Age"] = age
        out["Ownership_Age"] = ownership_age
        out["Km_Per_Year"] = km / np.maximum(age, 1.0)
        out["Km_Ratio"] = km / expected
        out["Delhi_Flag"] = out["Location"].isin(config.NCR_CITIES).astype(np.int8)
        out["Delhi_Restricted"] = (
            (out["Delhi_Flag"] == 1)
            & (
                ((out["Fuel_Type"] == "Diesel") & (age >= 10))
                | ((out["Fuel_Type"] == "Petrol") & (age >= 15))
            )
        ).astype(np.int8)

        for column, mapping in config.ORDINAL_MAPS.items():
            out[config.ORDINAL_OUTPUT_NAMES[column]] = (
                out[column].map(mapping).fillna(0).astype(np.int8)
            )

        out["Model_Key"] = (
            out["Manufacturer"].astype(str) + "|" + out["Model"].astype(str)
        )

        if self.add_interactions:
            condition = out["Current_Condition"].map(config.ORDINAL_MAPS["Current_Condition"]).fillna(0).astype(float)
            out["Age_Sq"] = age**2
            out["Age_x_Condition"] = age * condition
            for fuel in config.FUEL_LEVELS:
                out[f"Age_x_Fuel_{fuel}"] = age * (out["Fuel_Type"] == fuel).astype(float)
        return out


class ColumnSelect(BaseEstimator, TransformerMixin):
    """Select and order a fixed set of columns from a DataFrame."""

    def __init__(self, columns: list[str]) -> None:
        self.columns = columns

    def fit(self, X: pd.DataFrame, y: object = None) -> "ColumnSelect":
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = _as_frame(X)
        return X.loc[:, self.columns].copy()

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        return np.asarray(self.columns, dtype=object)


class CategoricalCaster(BaseEstimator, TransformerMixin):
    """Convert selected object columns to pandas categorical dtype.

    Categories are learned on the fitted data; unseen values become missing,
    which LightGBM handles natively.
    """

    def __init__(self, columns: list[str]) -> None:
        self.columns = columns

    def fit(self, X: pd.DataFrame, y: object = None) -> "CategoricalCaster":
        X = _as_frame(X)
        self.categories_ = {
            col: pd.Index(X[col].astype(str).unique()).sort_values()
            for col in self.columns
        }
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = _as_frame(X)
        out = X.copy()
        for col in self.columns:
            values = out[col].astype(str)
            out[col] = pd.Categorical(values, categories=self.categories_[col])
        return out

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        return np.asarray(list(self.feature_names_in_), dtype=object)


def _numeric_columns(add_interactions: bool, use_original_price: bool) -> list[str]:
    columns = list(config.NUMERIC_FEATURES)
    if add_interactions:
        columns += list(config.INTERACTION_FEATURES)
    if use_original_price:
        columns.append(config.OPTIONAL_PRICE_COLUMN)
    return columns


def build_preprocessor(
    model_name: str,
    use_original_price: bool = config.USE_ORIGINAL_PRICE,
    te_smooth: float = 10.0,
    add_interactions: bool | None = None,
) -> Pipeline:
    """Build the feature + preprocessing pipeline for a model family.

    Args:
        model_name: One of ``config.MODEL_NAMES``.
        use_original_price: Include the optional original-price leakage feature.
        te_smooth: Smoothing strength for sklearn target encoding.
        add_interactions: Override interaction-term inclusion (Ridge defaults on).

    Returns:
        A sklearn pipeline whose output is ready for the model estimator.
    """
    if model_name not in config.MODEL_NAMES:
        raise ValueError(f"Unknown model: {model_name}")
    if add_interactions is None:
        add_interactions = model_name == "ridge"

    features = FeatureEngineer(add_interactions=add_interactions)
    numeric_cols = _numeric_columns(add_interactions, use_original_price)

    if model_name in {"lightgbm", "catboost"}:
        columns = numeric_cols + list(config.CATEGORICAL_FEATURES)
        steps: list[tuple[str, object]] = [
            ("features", features),
            ("select", ColumnSelect(columns)),
        ]
        if model_name == "lightgbm":
            steps.append(("cast", CategoricalCaster(list(config.CATEGORICAL_FEATURES))))
        return Pipeline(steps)

    scale = model_name in {"ridge", "mlp"}
    transformers: list[tuple[str, object, list[str]]] = [
        (
            "numeric",
            StandardScaler() if scale else "passthrough",
            numeric_cols,
        ),
        (
            "low_card",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            list(config.LOW_CARDINALITY_CATEGORICALS),
        ),
        (
            "model_target",
            TargetEncoder(
                cv=KFold(n_splits=5, shuffle=True, random_state=config.SEED),
                smooth=float(te_smooth),
            ),
            ["Model_Key"],
        ),
    ]
    prep = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline([("features", features), ("prep", prep)])


def catboost_feature_spec(
    use_original_price: bool = config.USE_ORIGINAL_PRICE,
) -> tuple[list[str], list[int]]:
    """Return CatBoost column order and categorical feature indices."""
    numeric_cols = _numeric_columns(False, use_original_price)
    columns = numeric_cols + list(config.CATEGORICAL_FEATURES)
    cat_indices = list(range(len(numeric_cols), len(columns)))
    return columns, cat_indices


def transformed_feature_names(preprocessor: Pipeline) -> list[str]:
    """Return readable feature names after preprocessing."""
    prep = preprocessor.named_steps.get("prep")
    if prep is None:
        select = preprocessor.named_steps.get("select")
        if select is not None:
            return list(select.columns)
        return []
    return [str(name) for name in prep.get_feature_names_out()]
