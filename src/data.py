"""Data loading, validation, and reproducible train/validation/test splits."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config


class DataValidationError(ValueError):
    """Raised when the dataset violates required invariants."""


def load_raw_data(path: Path | str | None = None) -> pd.DataFrame:
    """Load the CSV dataset with basic type coercion.

    Args:
        path: Optional CSV path. Defaults to ``config.DATA_PATH``.

    Returns:
        Raw dataframe.
    """
    path = Path(path) if path is not None else config.DATA_PATH
    df = pd.read_csv(path)
    for col in ["Year_Purchased", "Manufacturing_Year", "Kilometres_Driven",
                "Number_of_Previous_Owners", "Engine_Displacement_cc"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("int64")
    for col in [config.TARGET, config.OPTIONAL_PRICE_COLUMN]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
    return df


def validate_data(df: pd.DataFrame) -> dict[str, Any]:
    """Validate the dataset and return a report.

    Raises:
        DataValidationError: If any check fails.
    """
    issues: list[str] = []
    report: dict[str, Any] = {"rows": int(len(df)), "columns": int(df.shape[1])}

    required = set(config.RAW_INPUT_COLUMNS) | {config.TARGET, "Car_ID"}
    missing = sorted(required - set(df.columns))
    if missing:
        issues.append(f"Missing required columns: {missing}")

    null_counts = df.isna().sum()
    null_counts = null_counts[null_counts > 0]
    report["null_counts"] = {str(k): int(v) for k, v in null_counts.items()}
    if not null_counts.empty:
        issues.append(f"Null values found: {report['null_counts']}")

    if "Car_ID" in df.columns:
        duplicate_ids = int(df["Car_ID"].duplicated().sum())
        report["duplicate_car_ids"] = duplicate_ids
        if duplicate_ids:
            issues.append(f"Duplicate Car_ID values: {duplicate_ids}")

    duplicate_rows = int(df.duplicated().sum())
    report["duplicate_full_rows"] = duplicate_rows
    if duplicate_rows:
        issues.append(f"Duplicate full rows: {duplicate_rows}")

    if config.TARGET in df.columns:
        non_positive_targets = int((df[config.TARGET] <= 0).sum())
        report["non_positive_resale_prices"] = non_positive_targets
        if non_positive_targets:
            issues.append(f"Non-positive resale prices: {non_positive_targets}")

    if {config.TARGET, config.OPTIONAL_PRICE_COLUMN}.issubset(df.columns):
        invalid = df[df[config.TARGET] >= df[config.OPTIONAL_PRICE_COLUMN]]
        report["resale_ge_original"] = int(len(invalid))
        if len(invalid):
            issues.append(f"Resale >= original purchase price: {len(invalid)}")

    if "Fuel_Type" in df.columns and "Engine_Displacement_cc" in df.columns:
        ev_bad = df[(df["Fuel_Type"] == "Electric") & (df["Engine_Displacement_cc"] != 0)]
        non_ev_bad = df[(df["Fuel_Type"] != "Electric") & (df["Engine_Displacement_cc"] <= 0)]
        report["ev_nonzero_cc"] = int(len(ev_bad))
        report["non_ev_zero_cc"] = int(len(non_ev_bad))
        if len(ev_bad):
            issues.append(f"EV rows with non-zero engine displacement: {len(ev_bad)}")
        if len(non_ev_bad):
            issues.append(f"Non-EV rows with zero displacement: {len(non_ev_bad)}")

    if {"Year_Purchased", "Manufacturing_Year"}.issubset(df.columns):
        bad_years = df[df["Year_Purchased"] < df["Manufacturing_Year"]]
        report["purchase_before_manufacture"] = int(len(bad_years))
        if len(bad_years):
            issues.append(f"Year_Purchased < Manufacturing_Year: {len(bad_years)}")

    if {"Fuel_Type", "Transmission"}.issubset(df.columns):
        combos = df[["Fuel_Type", "Transmission"]].drop_duplicates()
        impossible = []
        for fuel, transmission in combos.itertuples(index=False):
            allowed = config.ALLOWED_FUEL_TRANSMISSION.get(str(fuel))
            if allowed is None or transmission not in allowed:
                impossible.append((fuel, transmission))
        report["impossible_fuel_transmission"] = len(impossible)
        report["impossible_combos"] = [list(x) for x in impossible]
        if impossible:
            issues.append(f"Impossible fuel/transmission combinations: {impossible}")

    report["valid"] = not issues
    report["issues"] = issues
    print(json.dumps(report, indent=2))
    if issues:
        raise DataValidationError("Data validation failed:\n- " + "\n- ".join(issues))
    return report


def _price_bins(y: pd.Series, n_bins: int = config.N_PRICE_BINS) -> pd.Series:
    """Create stratified price bins, collapsing rare duplicate edges."""
    bins = pd.qcut(y, q=n_bins, labels=False, duplicates="drop")
    return bins.astype(int)


def make_splits(
    df: pd.DataFrame,
    force: bool = False,
    path: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create or load the canonical 70/15/15 split.

    The split is stratified by price quantile bins and saved so every notebook
    and the dashboard use identical rows.
    """
    config.ensure_dirs()
    path = path or config.SPLIT_PATH
    if path.exists() and not force:
        saved = np.load(path)
        train_idx = saved["train"]
        val_idx = saved["val"]
        test_idx = saved["test"]
        if len(np.unique(np.concatenate([train_idx, val_idx, test_idx]))) != len(df):
            raise ValueError("Saved split indices do not match the dataset length.")
        return train_idx, val_idx, test_idx

    y = df[config.TARGET]
    bins = _price_bins(y)
    all_idx = np.arange(len(df))
    train_idx, temp_idx = train_test_split(
        all_idx,
        test_size=config.VALIDATION_FRACTION + config.TEST_FRACTION,
        random_state=config.SEED,
        stratify=bins,
    )
    relative_test = config.TEST_FRACTION / (config.VALIDATION_FRACTION + config.TEST_FRACTION)
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=relative_test,
        random_state=config.SEED,
        stratify=bins.iloc[temp_idx],
    )
    np.savez_compressed(
        path,
        train=train_idx,
        val=val_idx,
        test=test_idx,
    )
    return train_idx, val_idx, test_idx


def split_frames(
    df: pd.DataFrame,
    force: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return train, validation, and test dataframes using saved indices."""
    train_idx, val_idx, test_idx = make_splits(df, force=force)
    train = df.iloc[train_idx].reset_index(drop=True)
    val = df.iloc[val_idx].reset_index(drop=True)
    test = df.iloc[test_idx].reset_index(drop=True)
    print(
        f"Split sizes -> train: {len(train):,} | val: {len(val):,} | test: {len(test):,}"
    )
    return train, val, test


def get_tuning_sample(
    train: pd.DataFrame,
    size: int = config.TUNING_SAMPLE_SIZE,
    force: bool = False,
) -> pd.DataFrame:
    """Return a stratified tuning subsample from the training rows only."""
    config.ensure_dirs()
    path = config.TUNING_INDICES_PATH
    n = min(size, len(train))
    if path.exists() and not force:
        idx = np.load(path)
        if len(idx) == n and len(idx) <= len(train):
            return train.iloc[idx].reset_index(drop=True)
    bins = _price_bins(train[config.TARGET])
    idx, _ = train_test_split(
        np.arange(len(train)),
        train_size=n,
        random_state=config.SEED,
        stratify=bins,
    )
    idx = np.sort(idx)
    np.save(path, idx)
    return train.iloc[idx].reset_index(drop=True)


def feature_target_split(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Split a dataframe into raw model inputs and rupee targets."""
    return df.drop(columns=[config.TARGET]), df[config.TARGET].to_numpy(dtype=float)


def prepare_training_frame(df: pd.DataFrame, use_original_price: bool = config.USE_ORIGINAL_PRICE) -> pd.DataFrame:
    """Drop identifiers, leakage columns, and the optional original price."""
    drop = [c for c in config.ID_COLUMNS + config.LEAKAGE_COLUMNS if c in df.columns]
    if not use_original_price and config.OPTIONAL_PRICE_COLUMN in df.columns:
        drop.append(config.OPTIONAL_PRICE_COLUMN)
    return df.drop(columns=drop)
