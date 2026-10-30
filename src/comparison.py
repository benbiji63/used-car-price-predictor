"""Analysis helpers for the model comparison notebook."""

from __future__ import annotations

import gc
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd

from . import config, metrics, training


def recompute_test_predictions(
    model_names: Iterable[str],
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    """Load each artifact, predict the test set, and return a prediction frame."""
    X_test = test.drop(columns=[config.TARGET])
    y_true = test[config.TARGET].to_numpy(float)
    frame = pd.DataFrame(
        {
            "y_true_inr": y_true,
            "y_true_log": np.log1p(y_true),
        }
    )
    recomputed: dict[str, dict[str, float]] = {}
    for name in model_names:
        artifact = joblib.load(config.model_path(name))
        pred_log = np.asarray(artifact["pipeline"].predict(X_test), dtype=float)
        frame[name] = pred_log
        recomputed[name] = metrics.evaluate_inr(y_true, pred_log)
        del artifact
        gc.collect()
    return frame, recomputed


def assert_metrics_match(
    recomputed: dict[str, dict[str, float]],
    saved: dict[str, Any],
    tolerance: float = 1e-6,
) -> pd.DataFrame:
    """Compare freshly recomputed metrics with metrics.json."""
    rows = []
    for name, fresh in recomputed.items():
        stored = saved.get(name, {}).get("test", {})
        for key, value in fresh.items():
            old = stored.get(key)
            diff = abs(float(old) - float(value)) if old is not None else float("nan")
            rows.append({"Model": name, "Metric": key, "metrics_json": old, "recomputed": value, "abs_diff": diff})
    frame = pd.DataFrame(rows)
    bad = frame[frame["abs_diff"].isna() | (frame["abs_diff"] > tolerance)]
    if len(bad):
        raise AssertionError(f"Metrics mismatch:\n{bad}")
    return frame


def leaderboard(pred_frame: pd.DataFrame, saved_metrics: dict[str, Any]) -> pd.DataFrame:
    """Build a test-set leaderboard from recomputed predictions and metadata."""
    rows = []
    for name in config.MODEL_NAMES:
        if name not in pred_frame.columns:
            continue
        test_metrics = metrics.evaluate_inr(
            pred_frame["y_true_inr"].to_numpy(float),
            pred_frame[name].to_numpy(float),
        )
        meta = saved_metrics.get(name, {})
        rows.append(
            {
                "Model": config.MODEL_LABELS[name],
                "key": name,
                "r2_log": test_metrics["r2_log"],
                "rmse_log": test_metrics["rmse_log"],
                "mae_inr": test_metrics["mae_inr"],
                "rmse_inr": test_metrics["rmse_inr"],
                "mape": test_metrics["mape"],
                "median_ape": test_metrics["median_ape"],
                "train_time_s": meta.get("train_time_s", np.nan),
                "inference_ms_per_1k": meta.get("inference_ms_per_1k", np.nan),
                "size_mb": meta.get("size_mb", np.nan),
            }
        )
    return pd.DataFrame(rows).sort_values("rmse_log").reset_index(drop=True)


def ensemble_predictions(
    pred_frame: pd.DataFrame,
    model_names: Iterable[str],
) -> np.ndarray:
    """Average selected model predictions in log space."""
    names = list(model_names)
    return pred_frame[names].to_numpy(float).mean(axis=1)


def evaluate_ensemble(
    pred_frame: pd.DataFrame,
    model_names: Iterable[str],
) -> dict[str, float]:
    """Evaluate an ensemble average on the test set."""
    log_pred = ensemble_predictions(pred_frame, model_names)
    return metrics.evaluate_inr(pred_frame["y_true_inr"].to_numpy(float), log_pred)


def ape_frame(pred_frame: pd.DataFrame, sample_size: int = 20_000, seed: int = config.SEED) -> pd.DataFrame:
    """Long-form absolute percentage errors for model comparison plots."""
    sample = pred_frame.sample(min(sample_size, len(pred_frame)), random_state=seed)
    y_true = sample["y_true_inr"].to_numpy(float)
    rows = []
    for name in config.MODEL_NAMES:
        if name not in sample.columns:
            continue
        pred = np.expm1(sample[name].to_numpy(float))
        ape = np.abs((y_true - pred) / np.maximum(y_true, 1.0)) * 100.0
        rows.append(pd.DataFrame({"Model": config.MODEL_LABELS[name], "APE": ape}))
    return pd.concat(rows, ignore_index=True)


def segment_error_frame(
    pred_frame: pd.DataFrame,
    test: pd.DataFrame,
    sample_size: int = 30_000,
    seed: int = config.SEED,
) -> pd.DataFrame:
    """Long-form APE by model and segment for the comparison chart."""
    sample = test.sample(min(sample_size, len(test)), random_state=seed)
    positions = sample.index.to_numpy()
    local = pred_frame.iloc[positions].reset_index(drop=True)
    test_local = sample.reset_index(drop=True)
    y_true = local["y_true_inr"].to_numpy(float)
    segments = pd.DataFrame(
        {
            "actual": y_true,
            "price_band": pd.qcut(y_true, 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"]),
            "age_band": pd.cut(
                config.REF_YEAR - test_local["Manufacturing_Year"].to_numpy(),
                bins=[-1, 2, 5, 8, 11, 14, 20],
                labels=["0-2", "3-5", "6-8", "9-11", "12-14", "15+"],
            ),
            "fuel": test_local["Fuel_Type"].to_numpy(),
        }
    )
    rows = []
    for name in config.MODEL_NAMES:
        if name not in local.columns:
            continue
        pred = np.expm1(local[name].to_numpy(float))
        frame = segments.copy()
        frame["APE"] = np.abs((y_true - pred) / np.maximum(y_true, 1.0)) * 100.0
        frame["Model"] = config.MODEL_LABELS[name]
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def residual_correlation(pred_frame: pd.DataFrame) -> pd.DataFrame:
    """Correlation matrix of log-scale residuals."""
    columns = {}
    for name in config.MODEL_NAMES:
        if name in pred_frame.columns:
            columns[config.MODEL_LABELS[name]] = pred_frame["y_true_log"] - pred_frame[name]
    return pd.DataFrame(columns).corr()


def run_ablation(
    best_model: str,
    params: dict[str, Any],
    train: pd.DataFrame,
    test: pd.DataFrame,
    baseline_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Refit the best model with Original_Purchase_Price_INR and evaluate."""
    best_iteration = params.get("best_iteration")
    pipeline = training.build_pipeline(
        best_model,
        params,
        use_original_price=True,
        final_iterations=int(best_iteration) if best_iteration else None,
    )
    X_train = train.drop(columns=[config.TARGET])
    y_train = np.log1p(train[config.TARGET].to_numpy(float))
    start = time.perf_counter()
    pipeline = training.fit_pipeline(pipeline, X_train, y_train)
    elapsed = time.perf_counter() - start
    pred_log = pipeline.predict(test.drop(columns=[config.TARGET]))
    test_metrics = metrics.evaluate_inr(test[config.TARGET].to_numpy(float), pred_log)
    payload = {
        "model_label": f"{config.MODEL_LABELS[best_model]} + original price",
        "base_model": best_model,
        "use_original_price": True,
        "test": test_metrics,
        "baseline_test": baseline_metrics,
        "train_time_s": float(elapsed),
        "note": "Leakage ablation only; never used as the dashboard winner.",
    }
    training.append_metrics("ablation_original_price", payload)
    return payload


def _load_params(model_name: str) -> dict[str, Any]:
    with open(config.params_path(model_name), encoding="utf-8") as handle:
        return json.load(handle)["params"]


def _fit_and_evaluate(
    model_name: str,
    params: dict[str, Any],
    fit_df: pd.DataFrame,
    eval_df: pd.DataFrame,
) -> dict[str, float]:
    best_iteration = params.get("best_iteration")
    pipeline = training.build_pipeline(
        model_name,
        params,
        final_iterations=int(best_iteration) if best_iteration else None,
    )
    pipeline = training.fit_pipeline(
        pipeline,
        fit_df.drop(columns=[config.TARGET]),
        np.log1p(fit_df[config.TARGET].to_numpy(float)),
    )
    pred_log = pipeline.predict(eval_df.drop(columns=[config.TARGET]))
    return metrics.evaluate_inr(eval_df[config.TARGET].to_numpy(float), pred_log)


def run_generalization(
    train: pd.DataFrame,
    test: pd.DataFrame,
    pred_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Hold out 15 car models, then hold out Manufacturing_Year >= 2024."""
    rng = np.random.default_rng(config.SEED)
    counts = train["Model"].value_counts()
    eligible = np.asarray(sorted(counts[counts >= 300].index.tolist()), dtype=object)
    holdout_models = list(rng.choice(eligible, size=min(15, len(eligible)), replace=False))

    car_rows: list[dict[str, Any]] = []
    year_rows: list[dict[str, Any]] = []
    for model_name in config.MODEL_NAMES:
        params = _load_params(model_name)

        train_mask = ~train["Model"].isin(holdout_models)
        eval_mask = test["Model"].isin(holdout_models)
        fit_df = train[train_mask]
        eval_df = test[eval_mask]
        fresh = _fit_and_evaluate(model_name, params, fit_df, eval_df)
        baseline = metrics.evaluate_inr(
            eval_df[config.TARGET].to_numpy(float),
            pred_frame.loc[eval_mask.to_numpy(), model_name].to_numpy(float),
        )
        car_rows.append(
            {
                "Model": config.MODEL_LABELS[model_name],
                "baseline_rmse_log": baseline["rmse_log"],
                "holdout_rmse_log": fresh["rmse_log"],
                "degradation_pct": (fresh["rmse_log"] / baseline["rmse_log"] - 1.0) * 100.0,
                "baseline_mape": baseline["mape"],
                "holdout_mape": fresh["mape"],
                "n_holdout": int(len(eval_df)),
            }
        )

        train_year_mask = train["Manufacturing_Year"] < 2024
        eval_year_mask = test["Manufacturing_Year"] >= 2024
        fit_year = train[train_year_mask]
        eval_year = test[eval_year_mask]
        fresh_year = _fit_and_evaluate(model_name, params, fit_year, eval_year)
        baseline_year = metrics.evaluate_inr(
            eval_year[config.TARGET].to_numpy(float),
            pred_frame.loc[eval_year_mask.to_numpy(), model_name].to_numpy(float),
        )
        year_rows.append(
            {
                "Model": config.MODEL_LABELS[model_name],
                "baseline_rmse_log": baseline_year["rmse_log"],
                "holdout_rmse_log": fresh_year["rmse_log"],
                "degradation_pct": (fresh_year["rmse_log"] / baseline_year["rmse_log"] - 1.0) * 100.0,
                "baseline_mape": baseline_year["mape"],
                "holdout_mape": fresh_year["mape"],
                "n_holdout": int(len(eval_year)),
            }
        )

    car_frame = pd.DataFrame(car_rows)
    year_frame = pd.DataFrame(year_rows)
    payload = {
        "holdout_car_models": holdout_models,
        "car_holdout": car_frame.to_dict(orient="records"),
        "year_holdout": year_frame.to_dict(orient="records"),
    }
    with open(config.ARTIFACTS_DIR / "generalization.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return car_frame, year_frame, holdout_models


def delhi_rule_check(
    test: pd.DataFrame,
    model_names: Iterable[str],
    per_age: int = 1_500,
) -> pd.DataFrame:
    """Counterfactual Delhi vs Pune price ratios for diesel age 9 and 11."""
    age = config.REF_YEAR - test["Manufacturing_Year"].astype(int)
    base = test[(test["Fuel_Type"] == "Diesel") & age.isin([9, 11])].copy()
    base["Age"] = config.REF_YEAR - base["Manufacturing_Year"].astype(int)
    samples = []
    for age_value, frame in base.groupby("Age"):
        samples.append(frame.sample(min(per_age, len(frame)), random_state=config.SEED))
    sample = pd.concat(samples, ignore_index=True)

    rows: list[dict[str, Any]] = []
    for model_name in model_names:
        artifact = joblib.load(config.model_path(model_name))
        pipeline = artifact["pipeline"]
        for age_value, frame in sample.groupby("Age"):
            ncr = frame.copy()
            non_ncr = frame.copy()
            ncr["Location"] = "Delhi"
            non_ncr["Location"] = "Pune"
            pred_ncr = np.expm1(pipeline.predict(ncr.drop(columns=[config.TARGET])))
            pred_non = np.expm1(pipeline.predict(non_ncr.drop(columns=[config.TARGET])))
            ratio = float(np.exp(np.mean(np.log(np.maximum(pred_ncr, 1.0)) - np.log(np.maximum(pred_non, 1.0)))))
            rows.append({"Series": config.MODEL_LABELS[model_name], "Age": int(age_value), "ratio": ratio})
        del artifact
        gc.collect()

    actual = base.copy()
    actual["ncr"] = actual["Location"].isin(config.NCR_CITIES)
    for age_value, frame in actual.groupby("Age"):
        ratio_ncr = (frame.loc[frame["ncr"], config.TARGET] / frame.loc[frame["ncr"], config.OPTIONAL_PRICE_COLUMN]).median()
        ratio_non = (frame.loc[~frame["ncr"], config.TARGET] / frame.loc[~frame["ncr"], config.OPTIONAL_PRICE_COLUMN]).median()
        rows.append(
            {
                "Series": "Actual data",
                "Age": int(age_value),
                "ratio": float(ratio_ncr / ratio_non),
            }
        )
    return pd.DataFrame(rows)


def choose_best(leaderboard_frame: pd.DataFrame) -> pd.Series:
    """Return the row with the lowest test log-RMSE."""
    return leaderboard_frame.sort_values("rmse_log").iloc[0]


def write_best_model(
    best_row: pd.Series,
    ensemble_metrics_all: dict[str, float],
    ensemble_metrics_top3: dict[str, float],
) -> dict[str, Any]:
    """Write artifacts/best_model.json for the dashboard."""
    payload = {
        "model_name": str(best_row["key"]),
        "model_label": str(best_row["Model"]),
        "selection_metric": "test.rmse_log",
        "test_metrics": {
            "r2_log": float(best_row["r2_log"]),
            "rmse_log": float(best_row["rmse_log"]),
            "mae_inr": float(best_row["mae_inr"]),
            "rmse_inr": float(best_row["rmse_inr"]),
            "mape": float(best_row["mape"]),
            "median_ape": float(best_row["median_ape"]),
        },
        "ensemble_average_test_metrics": ensemble_metrics_all,
        "ensemble_top3_test_metrics": ensemble_metrics_top3,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "note": "Selected from the six default models by test log-RMSE; leakage ablation excluded.",
    }
    with open(config.BEST_MODEL_PATH, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return payload
