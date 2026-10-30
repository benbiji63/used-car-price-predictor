"""Evaluation metrics and Indian currency formatting helpers."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def format_inr(value: float, decimals: int = 0) -> str:
    """Format a number using Indian digit grouping, e.g. ``₹6,12,000``.

    Args:
        value: Amount in rupees.
        decimals: Number of decimal places.

    Returns:
        Indian-grouped currency string.
    """
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "₹—"
    negative = value < 0
    text = f"{abs(float(value)):.{decimals}f}"
    integer, _, fraction = text.partition(".")
    if len(integer) <= 3:
        grouped = integer
    else:
        last_three = integer[-3:]
        rest = integer[:-3]
        parts: list[str] = []
        while len(rest) > 2:
            parts.insert(0, rest[-2:])
            rest = rest[:-2]
        if rest:
            parts.insert(0, rest)
        grouped = ",".join(parts + [last_three])
    result = f"₹{grouped}"
    if fraction:
        result += f".{fraction}"
    if negative:
        result = "-" + result
    return result


def compact_inr(value: float) -> str:
    """Compact Indian currency label for chart axes."""
    value = float(value)
    if abs(value) >= 1e7:
        return f"₹{value / 1e7:,.2f} Cr"
    if abs(value) >= 1e5:
        return f"₹{value / 1e5:,.1f} L"
    if abs(value) >= 1e3:
        return f"₹{value / 1e3:,.1f}k"
    return f"₹{value:,.0f}"


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute percentage error in percent."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), 1.0)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def median_ape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Median absolute percentage error in percent."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), 1.0)
    return float(np.median(np.abs((y_true - y_pred) / denom)) * 100.0)


def calibration_factor(
    y_true_inr: np.ndarray,
    pred_log: np.ndarray,
    method: str = "log_mean",
) -> float:
    """Estimate a multiplicative correction for systematic prediction bias.

    Args:
        y_true_inr: Actual resale prices in INR.
        pred_log: Predictions on the ``log1p`` scale.
        method: ``"log_mean"`` (mean log residual), ``"ratio_mean"``,
            ``"ratio_median"``, or ``"total"`` (sum of actual / sum of predicted).

    Returns:
        A positive factor that can be applied multiplicatively in INR space or
        as ``pred_log + log(factor)`` in log space.
    """
    y_true = np.asarray(y_true_inr, dtype=float)
    pred_inr = np.maximum(np.expm1(np.asarray(pred_log, dtype=float)), 1.0)
    if method == "log_mean":
        factor = float(np.exp(np.mean(np.log1p(y_true) - np.asarray(pred_log, dtype=float))))
    elif method == "ratio_mean":
        factor = float(np.mean(y_true / pred_inr))
    elif method == "ratio_median":
        factor = float(np.median(y_true / pred_inr))
    elif method == "total":
        factor = float(y_true.sum() / pred_inr.sum())
    else:
        raise ValueError(f"Unknown calibration method: {method}")
    return max(factor, 1e-6)


def signed_bias(y_true_inr: np.ndarray, pred_log: np.ndarray) -> dict[str, float]:
    """Measure under/over-prediction and suggest multiplicative corrections.

    Positive percentages mean the model over-predicts; negative means it
    under-predicts. Ratios are ``predicted / actual``.
    """
    y_true = np.asarray(y_true_inr, dtype=float)
    pred_inr = np.maximum(np.expm1(np.asarray(pred_log, dtype=float)), 0.0)
    denom = np.maximum(np.abs(y_true), 1.0)
    signed_pct = (pred_inr - y_true) / denom
    ratio = pred_inr / denom
    return {
        "mean_signed_pct": float(np.mean(signed_pct) * 100.0),
        "median_signed_pct": float(np.median(signed_pct) * 100.0),
        "mean_ratio": float(np.mean(ratio)),
        "median_ratio": float(np.median(ratio)),
        "underprediction_rate": float(np.mean(pred_inr < y_true) * 100.0),
        "calibration_factor_log": calibration_factor(y_true, pred_log, "log_mean"),
        "calibration_factor_inr": calibration_factor(y_true, pred_log, "total"),
    }


def evaluate_inr(y_true_inr: np.ndarray, pred_log: np.ndarray) -> dict[str, float]:
    """Evaluate log-space predictions against rupee targets.

    Args:
        y_true_inr: Actual resale prices in INR.
        pred_log: Predictions on the ``log1p`` scale.

    Returns:
        Dictionary with log and INR-space metrics.
    """
    y_true_inr = np.asarray(y_true_inr, dtype=float)
    pred_log = np.asarray(pred_log, dtype=float)
    y_log = np.log1p(y_true_inr)
    pred_inr = np.maximum(np.expm1(pred_log), 0.0)
    return {
        "r2_log": float(r2_score(y_log, pred_log)),
        "rmse_log": float(math.sqrt(mean_squared_error(y_log, pred_log))),
        "mae_log": float(mean_absolute_error(y_log, pred_log)),
        "mae_inr": float(mean_absolute_error(y_true_inr, pred_inr)),
        "rmse_inr": float(math.sqrt(mean_squared_error(y_true_inr, pred_inr))),
        "mape": mape(y_true_inr, pred_inr),
        "median_ape": median_ape(y_true_inr, pred_inr),
    }


def evaluate_df(y_true_inr: np.ndarray, pred_log: np.ndarray) -> dict[str, float]:
    """Alias kept explicit for notebook readability."""
    return evaluate_inr(y_true_inr, pred_log)


def metric_table(metrics: dict[str, float]) -> dict[str, str]:
    """Return display-friendly formatted metric values."""
    return {
        "R² (log)": f"{metrics['r2_log']:.5f}",
        "RMSE (log)": f"{metrics['rmse_log']:.5f}",
        "MAE (log)": f"{metrics['mae_log']:.5f}",
        "MAE (INR)": format_inr(metrics["mae_inr"]),
        "RMSE (INR)": format_inr(metrics["rmse_inr"]),
        "MAPE": f"{metrics['mape']:.2f}%",
        "Median APE": f"{metrics['median_ape']:.2f}%",
    }


def bootstrap_rmse_difference(
    y_true_inr: np.ndarray,
    pred_log_a: np.ndarray,
    pred_log_b: np.ndarray,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    """Paired bootstrap confidence interval for RMSE(A) - RMSE(B).

    Uses log-space squared errors and resamples rows with replacement.
    """
    rng = np.random.default_rng(seed)
    y_log = np.log1p(np.asarray(y_true_inr, dtype=float))
    err_a = (np.asarray(pred_log_a, dtype=float) - y_log) ** 2
    err_b = (np.asarray(pred_log_b, dtype=float) - y_log) ** 2
    n = len(y_log)
    diffs = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        diffs[i] = math.sqrt(err_a[idx].mean()) - math.sqrt(err_b[idx].mean())
    point = math.sqrt(err_a.mean()) - math.sqrt(err_b.mean())
    return {
        "rmse_a": float(math.sqrt(err_a.mean())),
        "rmse_b": float(math.sqrt(err_b.mean())),
        "difference": float(point),
        "ci_low": float(np.percentile(diffs, 2.5)),
        "ci_high": float(np.percentile(diffs, 97.5)),
        "n_bootstrap": int(n_bootstrap),
        "probability_a_better": float(np.mean(diffs < 0)),
    }


def metrics_row(name: str, metrics: dict[str, float]) -> dict[str, Any]:
    """Flatten metrics for a comparison dataframe."""
    row: dict[str, Any] = {"Model": name}
    row.update(metrics)
    return row
