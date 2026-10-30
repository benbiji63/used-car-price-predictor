"""Artifact loading and prediction helpers for notebooks and the dashboard."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd

from . import bias, config, metrics


@lru_cache(maxsize=16)
def load_artifact(model_name: str) -> dict[str, Any]:
    """Load a serialized model artifact.

    Args:
        model_name: Registered model key, e.g. ``"lightgbm"``.

    Returns:
        Artifact dictionary containing the raw-data pipeline.
    """
    path = config.model_path(model_name)
    if not path.exists():
        raise FileNotFoundError(f"Missing artifact {path}. Run the corresponding notebook first.")
    return joblib.load(path)


def available_models() -> list[str]:
    """Return model names with serialized artifacts present."""
    return [name for name in config.MODEL_NAMES if config.model_path(name).exists()]


def _ensure_input_frame(data: pd.DataFrame | dict[str, Any] | list[dict[str, Any]]) -> pd.DataFrame:
    if isinstance(data, dict):
        frame = pd.DataFrame([data])
    elif isinstance(data, list):
        frame = pd.DataFrame(data)
    else:
        frame = data.copy()
    missing = [col for col in config.RAW_INPUT_COLUMNS if col not in frame.columns]
    if missing:
        raise ValueError(f"Prediction input is missing columns: {missing}")
    return frame


@lru_cache(maxsize=1)
def load_calibration() -> dict[str, Any]:
    """Load saved calibration factors, if the calibration step has been run."""
    return bias.load_calibration()


def calibration_factor(model_name: str) -> float:
    """Return the saved multiplicative factor for a model, or 1.0."""
    return bias.calibration_factors(load_calibration()).get(model_name, 1.0)


def predict_log(
    model_name: str,
    data: pd.DataFrame | dict[str, Any] | list[dict[str, Any]],
    *,
    calibrated: bool = False,
) -> np.ndarray:
    """Predict log1p(resale price) for raw input rows."""
    frame = _ensure_input_frame(data)
    artifact = load_artifact(model_name)
    if artifact.get("use_original_price") and config.OPTIONAL_PRICE_COLUMN not in frame:
        raise ValueError("This artifact requires Original_Purchase_Price_INR.")
    pred_log = np.asarray(artifact["pipeline"].predict(frame), dtype=float)
    if calibrated:
        pred_log = bias.apply_factor(pred_log, calibration_factor(model_name))
    return pred_log


def predict_inr(
    model_name: str,
    data: pd.DataFrame | dict[str, Any] | list[dict[str, Any]],
    *,
    calibrated: bool = False,
) -> np.ndarray:
    """Predict resale price in INR for raw input rows."""
    return np.maximum(np.expm1(predict_log(model_name, data, calibrated=calibrated)), 0.0)


def predict_ensemble(
    data: pd.DataFrame | dict[str, Any] | list[dict[str, Any]],
    model_names: Iterable[str] | None = None,
    weights: Iterable[float] | None = None,
    *,
    return_log: bool = False,
    calibrated: bool = False,
) -> np.ndarray:
    """Average model predictions in log space, then convert to INR.

    Averaging on the log scale is equivalent to a geometric mean of prices,
    which is more stable for this multiplicative price generator.
    """
    names = list(model_names) if model_names is not None else available_models()
    if not names:
        raise ValueError("No models selected for the ensemble.")
    logs = np.vstack(
        [predict_log(name, data, calibrated=calibrated) for name in names]
    )
    if weights is None:
        log_pred = logs.mean(axis=0)
    else:
        weight_array = np.asarray(list(weights), dtype=float)
        if len(weight_array) != len(names):
            raise ValueError("Number of weights must match number of models.")
        weight_array = weight_array / weight_array.sum()
        log_pred = np.average(logs, axis=0, weights=weight_array)
    return log_pred if return_log else np.maximum(np.expm1(log_pred), 0.0)


def prediction_range(
    pred_inr: float | np.ndarray,
    rmse_log: float,
    z: float = 1.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a log-scale RMSE into a multiplicative INR prediction interval."""
    pred = np.asarray(pred_inr, dtype=float)
    factor = float(np.exp(z * rmse_log))
    return pred / factor, pred * factor


def read_best_model() -> dict[str, Any]:
    """Read the comparison notebook's best-model decision."""
    if not config.BEST_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing {config.BEST_MODEL_PATH}. Run notebooks/07_model_comparison.ipynb first."
        )
    with open(config.BEST_MODEL_PATH, encoding="utf-8") as handle:
        return json.load(handle)


def best_model_name() -> str:
    """Return the winning model key from ``best_model.json``."""
    return str(read_best_model()["model_name"])


def load_metrics() -> dict[str, Any]:
    """Load the shared metrics JSON."""
    if not config.METRICS_PATH.exists():
        return {}
    with open(config.METRICS_PATH, encoding="utf-8") as handle:
        return json.load(handle)


def load_bias_report() -> dict[str, Any]:
    """Load the saved signed-bias report, if present."""
    if not config.BIAS_REPORT_PATH.exists():
        return {}
    with open(config.BIAS_REPORT_PATH, encoding="utf-8") as handle:
        return json.load(handle)


def load_catalog(path: Path | None = None) -> dict[str, Any]:
    """Load the dashboard catalog options JSON."""
    path = path or config.CATALOG_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run `python -m src.build_catalog_options` first."
        )
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def load_test_predictions() -> pd.DataFrame | None:
    """Load saved per-model test predictions when available."""
    if not config.TEST_PREDICTIONS_PATH.exists():
        return None
    return pd.read_parquet(config.TEST_PREDICTIONS_PATH)


def ensemble_metrics_from_test(
    model_names: Iterable[str],
    test_predictions: pd.DataFrame | None = None,
) -> dict[str, float] | None:
    """Compute ensemble metrics for an arbitrary subset of models."""
    frame = test_predictions if test_predictions is not None else load_test_predictions()
    if frame is None:
        return None
    names = list(model_names)
    columns = [name for name in names if name in frame.columns]
    if not columns:
        return None
    log_pred = frame[columns].to_numpy(dtype=float).mean(axis=1)
    return metrics.evaluate_inr(frame["y_true_inr"].to_numpy(dtype=float), log_pred)


def model_metadata(model_name: str) -> dict[str, Any]:
    """Return metrics metadata for a model, or an empty dict."""
    return load_metrics().get(model_name, {})
