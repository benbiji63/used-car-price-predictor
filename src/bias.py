"""Signed-bias diagnostics and calibration helpers for saved models."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from . import config, metrics

SEGMENT_COLUMNS = [
    "Fuel_Type",
    "Body_Type",
    "Location",
    "Manufacturer",
    "Current_Condition",
    "Accident_History",
]


def apply_factor(pred_log: np.ndarray, factor: float) -> np.ndarray:
    """Apply a multiplicative calibration factor on the log1p scale."""
    return np.asarray(pred_log, dtype=float) + np.log(max(float(factor), 1e-6))


def _segments(frame: pd.DataFrame, y_true_inr: np.ndarray) -> pd.DataFrame:
    age = config.REF_YEAR - frame["Manufacturing_Year"].astype(int)
    segments = pd.DataFrame(
        {
            "price_band": pd.qcut(
                y_true_inr,
                5,
                labels=["Q1", "Q2", "Q3", "Q4", "Q5"],
                duplicates="drop",
            ),
            "age_band": pd.cut(
                age,
                bins=[-1, 2, 5, 8, 11, 14, 20],
                labels=["0-2", "3-5", "6-8", "9-11", "12-14", "15+"],
            ),
        }
    )
    for column in SEGMENT_COLUMNS:
        if column in frame.columns:
            segments[column] = frame[column].astype(str).to_numpy()
    return segments


def bias_by_segment(
    y_true_inr: np.ndarray,
    pred_log: np.ndarray,
    segment: pd.Series,
    min_rows: int = 30,
) -> dict[str, dict[str, float | int]]:
    """Return signed-bias statistics for each level of one segment column."""
    frame = pd.DataFrame(
        {
            "segment": segment.to_numpy(),
            "y_true": np.asarray(y_true_inr, dtype=float),
            "pred_log": np.asarray(pred_log, dtype=float),
        }
    )
    rows: dict[str, dict[str, float | int]] = {}
    for level, group in frame.groupby("segment", observed=True):
        if len(group) < min_rows:
            continue
        stats = metrics.signed_bias(
            group["y_true"].to_numpy(float), group["pred_log"].to_numpy(float)
        )
        rows[str(level)] = {
            "n": int(len(group)),
            "mean_signed_pct": stats["mean_signed_pct"],
            "median_signed_pct": stats["median_signed_pct"],
            "mean_ratio": stats["mean_ratio"],
            "underprediction_rate": stats["underprediction_rate"],
        }
    return rows


def build_bias_report(
    pred_frame: pd.DataFrame,
    reference: pd.DataFrame,
    model_names: Iterable[str] | None = None,
    source: str = "canonical test split",
    min_rows: int = 30,
) -> dict[str, Any]:
    """Build global and segmented signed-bias diagnostics.

    Args:
        pred_frame: Frame with ``y_true_inr`` and one log-prediction column per model.
        reference: Raw feature rows aligned with ``pred_frame`` for segmentation.
        model_names: Optional subset of models. Defaults to all present model columns.
        source: Human-readable description of the reference data.
    """
    if len(pred_frame) != len(reference):
        raise ValueError("pred_frame and reference must have the same number of rows.")
    if "y_true_inr" not in pred_frame.columns:
        raise ValueError("pred_frame must contain a y_true_inr column.")

    pred_frame = pred_frame.reset_index(drop=True)
    reference = reference.reset_index(drop=True)
    names = (
        list(model_names)
        if model_names is not None
        else [name for name in config.MODEL_NAMES if name in pred_frame.columns]
    )
    y_true = pred_frame["y_true_inr"].to_numpy(float)
    segments = _segments(reference, y_true)

    global_rows: dict[str, dict[str, float | int]] = {}
    segment_rows: dict[str, dict[str, Any]] = {}
    for name in names:
        if name not in pred_frame.columns:
            continue
        pred_log = pred_frame[name].to_numpy(float)
        stats = metrics.signed_bias(y_true, pred_log)
        stats.update(
            {
                "model_label": config.MODEL_LABELS.get(name, name),
                "n": int(len(y_true)),
            }
        )
        global_rows[name] = stats
        segment_rows[name] = {
            column: bias_by_segment(y_true, pred_log, segments[column], min_rows=min_rows)
            for column in segments.columns
        }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "rows": int(len(y_true)),
        "global": global_rows,
        "segments": segment_rows,
    }


def fit_calibration(
    pred_frame: pd.DataFrame,
    model_names: Iterable[str] | None = None,
    method: str = "log_mean",
    source: str = "canonical validation split",
) -> dict[str, Any]:
    """Fit per-model multiplicative calibration factors on labeled predictions."""
    if "y_true_inr" not in pred_frame.columns:
        raise ValueError("pred_frame must contain a y_true_inr column.")
    names = (
        list(model_names)
        if model_names is not None
        else [name for name in config.MODEL_NAMES if name in pred_frame.columns]
    )
    y_true = pred_frame["y_true_inr"].to_numpy(float)
    factors: dict[str, float] = {}
    model_metrics: dict[str, dict[str, Any]] = {}
    for name in names:
        if name not in pred_frame.columns:
            continue
        pred_log = pred_frame[name].to_numpy(float)
        factor = metrics.calibration_factor(y_true, pred_log, method)
        factors[name] = factor
        corrected = apply_factor(pred_log, factor)
        model_metrics[name] = {
            "model_label": config.MODEL_LABELS.get(name, name),
            "factor": factor,
            "before": metrics.evaluate_inr(y_true, pred_log),
            "after": metrics.evaluate_inr(y_true, corrected),
            "bias_before": metrics.signed_bias(y_true, pred_log),
            "bias_after": metrics.signed_bias(y_true, corrected),
        }
    return {
        "version": 1,
        "method": method,
        "source": source,
        "rows": int(len(y_true)),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "factors": factors,
        "metrics": model_metrics,
    }


def write_json(payload: dict[str, Any], path: Path) -> None:
    """Write a diagnostics payload as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)


def load_calibration(path: Path | None = None) -> dict[str, Any]:
    """Load saved calibration factors, returning an empty payload if absent."""
    path = path or config.CALIBRATION_PATH
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def calibration_factors(payload: dict[str, Any] | None = None) -> dict[str, float]:
    """Return only the model-to-factor mapping from a calibration payload."""
    payload = payload if payload is not None else load_calibration()
    return {
        str(name): float(value)
        for name, value in payload.get("factors", {}).items()
    }
