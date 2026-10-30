"""Shared model construction, Optuna tuning, final fitting, and persistence."""

from __future__ import annotations

import copy
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from . import config, features, metrics

optuna.logging.set_verbosity(optuna.logging.WARNING)


def jsonable(value: Any) -> Any:
    """Recursively convert numpy/tuple values into JSON-safe objects."""
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _split_tuning_frames(
    sample: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the tuning sample into fit / early-stop / objective partitions."""
    bins = pd.qcut(sample[config.TARGET], q=config.N_PRICE_BINS, labels=False, duplicates="drop")
    idx = np.arange(len(sample))
    fit_idx, rest_idx = train_test_split(
        idx,
        test_size=0.30,
        random_state=config.SEED,
        stratify=bins,
    )
    es_idx, val_idx = train_test_split(
        rest_idx,
        test_size=0.50,
        random_state=config.SEED,
        stratify=bins.iloc[rest_idx],
    )
    return (
        sample.iloc[fit_idx].reset_index(drop=True),
        sample.iloc[es_idx].reset_index(drop=True),
        sample.iloc[val_idx].reset_index(drop=True),
    )


def _ridge_alphas(params: dict[str, Any]) -> np.ndarray:
    lo = float(params.pop("alpha_min", 1e-4))
    hi = float(params.pop("alpha_max", 1e3))
    if hi <= lo:
        hi = lo * 10.0
    return np.logspace(math.log10(lo), math.log10(hi), 25)


def make_estimator(
    model_name: str,
    params: dict[str, Any],
    *,
    tuning: bool = False,
) -> Any:
    """Construct the underlying regressor from a parameter dictionary."""
    p = dict(params)
    if model_name == "ridge":
        return RidgeCV(alphas=_ridge_alphas(p), cv=5)

    if model_name == "random_forest":
        return RandomForestRegressor(
            n_estimators=int(p.get("n_estimators", 300)),
            max_depth=p.get("max_depth", 24),
            min_samples_leaf=int(p.get("min_samples_leaf", 2)),
            max_features=p.get("max_features", 0.6),
            n_jobs=8,
            random_state=config.SEED,
        )

    if model_name == "xgboost":
        kwargs: dict[str, Any] = dict(
            n_estimators=int(p.get("n_estimators", 800)),
            learning_rate=float(p.get("learning_rate", 0.05)),
            max_depth=int(p.get("max_depth", 8)),
            min_child_weight=float(p.get("min_child_weight", 3)),
            subsample=float(p.get("subsample", 0.85)),
            colsample_bytree=float(p.get("colsample_bytree", 0.85)),
            reg_lambda=float(p.get("reg_lambda", 1.0)),
            reg_alpha=float(p.get("reg_alpha", 0.0)),
            gamma=float(p.get("gamma", 0.0)),
            tree_method="hist",
            n_jobs=4,
            random_state=config.SEED,
            verbosity=0,
            eval_metric="rmse",
        )
        if tuning:
            kwargs["early_stopping_rounds"] = 50
        return XGBRegressor(**kwargs)

    if model_name == "lightgbm":
        return LGBMRegressor(
            n_estimators=int(p.get("n_estimators", 900)),
            learning_rate=float(p.get("learning_rate", 0.05)),
            num_leaves=int(p.get("num_leaves", 63)),
            max_depth=int(p.get("max_depth", -1)),
            min_child_samples=int(p.get("min_child_samples", 20)),
            subsample=float(p.get("subsample", 0.85)),
            subsample_freq=1 if float(p.get("subsample", 0.85)) < 1.0 else 0,
            colsample_bytree=float(p.get("colsample_bytree", 0.85)),
            reg_lambda=float(p.get("reg_lambda", 1.0)),
            reg_alpha=float(p.get("reg_alpha", 0.0)),
            n_jobs=4,
            random_state=config.SEED,
            verbose=-1,
            metric="rmse",
        )

    if model_name == "catboost":
        _, cat_indices = features.catboost_feature_spec(
            use_original_price=p.pop("_use_original_price", False)
        )
        kwargs = dict(
            iterations=int(p.get("iterations", 900)),
            learning_rate=float(p.get("learning_rate", 0.05)),
            depth=int(p.get("depth", 7)),
            l2_leaf_reg=float(p.get("l2_leaf_reg", 3.0)),
            random_strength=float(p.get("random_strength", 1.0)),
            bagging_temperature=float(p.get("bagging_temperature", 1.0)),
            border_count=int(p.get("border_count", 128)),
            cat_features=cat_indices,
            thread_count=8,
            random_seed=config.SEED,
            verbose=0,
            allow_writing_files=False,
            loss_function="RMSE",
            max_ctr_complexity=1,
            one_hot_max_size=8,
        )
        if tuning:
            kwargs["od_type"] = "Iter"
            kwargs["od_wait"] = 30
        return CatBoostRegressor(**kwargs)

    if model_name == "mlp":
        hidden = p.get("hidden_layer_sizes", (256, 128))
        if isinstance(hidden, list):
            hidden = tuple(int(x) for x in hidden)
        return MLPRegressor(
            hidden_layer_sizes=hidden,
            activation=str(p.get("activation", "relu")),
            alpha=float(p.get("alpha", 1e-4)),
            learning_rate_init=float(p.get("learning_rate_init", 1e-3)),
            batch_size=int(p.get("batch_size", 512)),
            max_iter=int(p.get("max_iter", 140)),
            early_stopping=True,
            validation_fraction=0.10,
            n_iter_no_change=8,
            random_state=config.SEED,
        )

    raise ValueError(f"Unknown model: {model_name}")


def build_pipeline(
    model_name: str,
    params: dict[str, Any],
    *,
    use_original_price: bool = False,
    tuning: bool = False,
    final_iterations: int | None = None,
) -> Pipeline:
    """Build a complete raw-data pipeline for one model."""
    p = dict(params)
    te_smooth = float(p.pop("te_smooth", 10.0))
    add_interactions = p.pop("add_interactions", None)
    p.pop("best_iteration", None)
    preprocessor = features.build_preprocessor(
        model_name,
        use_original_price=use_original_price,
        te_smooth=te_smooth,
        add_interactions=add_interactions,
    )
    if model_name in {"xgboost", "lightgbm"} and final_iterations is not None:
        p["n_estimators"] = int(final_iterations)
    if model_name == "catboost" and final_iterations is not None:
        p["iterations"] = int(final_iterations)
    if model_name == "mlp" and not tuning:
        p["max_iter"] = max(int(p.get("max_iter", 140)), 200)
    p["_use_original_price"] = use_original_price
    estimator = make_estimator(model_name, p, tuning=tuning)
    steps = list(preprocessor.steps) + [("model", estimator)]
    return Pipeline(steps)


def fit_pipeline(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train_log: np.ndarray,
    X_eval: pd.DataFrame | None = None,
    y_eval_log: np.ndarray | None = None,
    *,
    tuning: bool = False,
) -> Pipeline:
    """Fit a pipeline, optionally using an evaluation set for early stopping."""
    steps = list(pipeline.steps)
    feature_step = clone(steps[0][1])
    feature_step.fit(X_train)
    Xt = feature_step.transform(X_train)
    fitted_intermediate: list[tuple[str, object]] = []
    for name, step in steps[1:-1]:
        fitted_step = clone(step)
        fitted_step.fit(Xt, y_train_log)
        Xt = fitted_step.transform(Xt)
        fitted_intermediate.append((name, fitted_step))
    Xtt = Xt
    estimator = copy.deepcopy(steps[-1][1])

    Xe = None
    if X_eval is not None and y_eval_log is not None:
        Xe = feature_step.transform(X_eval)
        for _, fitted_step in fitted_intermediate:
            Xe = fitted_step.transform(Xe)

    model_name = pipeline.named_steps["model"].__class__.__name__
    if model_name == "LGBMRegressor":
        callbacks = (
            [early_stopping(50, verbose=False), log_evaluation(0)]
            if Xe is not None
            else None
        )
        estimator.fit(
            Xtt,
            y_train_log,
            eval_set=[(Xe, y_eval_log)] if Xe is not None else None,
            callbacks=callbacks,
        )
    elif model_name == "XGBRegressor":
        estimator.fit(
            Xtt,
            y_train_log,
            eval_set=[(Xe, y_eval_log)] if Xe is not None else None,
            verbose=False,
        )
    elif model_name == "CatBoostRegressor":
        estimator.fit(
            Xtt,
            y_train_log,
            eval_set=(Xe, y_eval_log) if Xe is not None else None,
            use_best_model=Xe is not None,
        )
    else:
        estimator.fit(Xtt, y_train_log)

    pipeline.steps = [("features", feature_step)] + fitted_intermediate + [("model", estimator)]
    return pipeline


def _best_iteration(estimator: Any) -> int | None:
    """Extract the early-stopping best iteration from a fitted estimator."""
    for attr in ("best_iteration", "best_iteration_"):
        value = getattr(estimator, attr, None)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                pass
    if hasattr(estimator, "get_best_iteration"):
        try:
            value = int(estimator.get_best_iteration())
            if value > 0:
                return value
        except Exception:
            pass
    return None


def _suggest_params(trial: optuna.Trial, model_name: str) -> dict[str, Any]:
    """Suggest hyperparameters for one Optuna trial."""
    if model_name == "ridge":
        return {
            "add_interactions": trial.suggest_categorical("add_interactions", [True, False]),
            "te_smooth": trial.suggest_float("te_smooth", 1.0, 50.0, log=True),
            "alpha_min": trial.suggest_float("alpha_min", 1e-5, 1e-1, log=True),
            "alpha_max": trial.suggest_float("alpha_max", 1.0, 1e4, log=True),
        }
    if model_name == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 150, 300, step=50),
            "max_depth": trial.suggest_int("max_depth", 10, 28),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 10),
            "max_features": trial.suggest_float("max_features", 0.3, 0.8),
            "te_smooth": trial.suggest_float("te_smooth", 1.0, 50.0, log=True),
        }
    if model_name == "xgboost":
        return {
            "n_estimators": 2500,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "min_child_weight": trial.suggest_float("min_child_weight", 0.5, 30.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 20.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "te_smooth": trial.suggest_float("te_smooth", 1.0, 50.0, log=True),
        }
    if model_name == "lightgbm":
        return {
            "n_estimators": 2500,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 255),
            "max_depth": trial.suggest_int("max_depth", 4, 18),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 60),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 20.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        }
    if model_name == "catboost":
        return {
            "iterations": 2000,
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "depth": trial.suggest_int("depth", 4, 9),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 20.0, log=True),
            "random_strength": trial.suggest_float("random_strength", 0.0, 2.0),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "border_count": trial.suggest_categorical("border_count", [64, 128]),
        }
    if model_name == "mlp":
        return {
            "hidden_layer_sizes": trial.suggest_categorical(
                "hidden_layer_sizes",
                [(128, 64), (256, 128), (256, 128, 64), (512, 256, 128)],
            ),
            "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
            "alpha": trial.suggest_float("alpha", 1e-6, 1e-2, log=True),
            "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 5e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [256, 512, 1024]),
            "te_smooth": trial.suggest_float("te_smooth", 1.0, 50.0, log=True),
            "max_iter": 60,
        }
    raise ValueError(f"Unknown model: {model_name}")


def _study_from_trial_frame(trial_df: pd.DataFrame) -> optuna.Study:
    """Rehydrate an Optuna study from a saved trials dataframe."""
    study = optuna.create_study(direction="minimize")
    param_cols = [c for c in trial_df.columns if c.startswith("params_")]
    for _, row in trial_df.iterrows():
        if str(row.get("state", "COMPLETE")) != "COMPLETE":
            continue
        params: dict[str, Any] = {}
        distributions: dict[str, optuna.distributions.BaseDistribution] = {}
        for col in param_cols:
            name = col.replace("params_", "", 1)
            value = row[col]
            if pd.isna(value):
                continue
            if isinstance(value, (bool, np.bool_)) or isinstance(value, str):
                value = bool(value) if isinstance(value, (bool, np.bool_)) else str(value)
                params[name] = value
                distributions[name] = optuna.distributions.CategoricalDistribution([value])
            elif float(value).is_integer():
                value = int(value)
                params[name] = value
                distributions[name] = optuna.distributions.IntDistribution(0, 1_000_000_000, step=1)
            else:
                value = float(value)
                params[name] = value
                distributions[name] = optuna.distributions.FloatDistribution(-1_000_000_000.0, 1_000_000_000.0)
        best_iter = row.get("user_attrs_best_iteration")
        user_attrs = {}
        if pd.notna(best_iter):
            user_attrs["best_iteration"] = int(best_iter)
        study.add_trial(
            optuna.trial.create_trial(
                value=float(row["value"]),
                params=params,
                distributions=distributions,
                user_attrs=user_attrs,
                state=optuna.trial.TrialState.COMPLETE,
            )
        )
    return study


def tune_model(
    model_name: str,
    tuning_sample: pd.DataFrame,
    n_trials: int,
    *,
    use_original_price: bool = False,
) -> tuple[optuna.Study, dict[str, Any], dict[str, float], dict[str, Any]]:
    """Tune one model on a train-only subsample.

    Returns:
        Study, best parameters, baseline metrics, and tuning metadata.
    """
    fit_df, es_df, val_df = _split_tuning_frames(tuning_sample)
    X_fit, y_fit = fit_df.drop(columns=[config.TARGET]), np.log1p(fit_df[config.TARGET].to_numpy(float))
    X_es, y_es = es_df.drop(columns=[config.TARGET]), np.log1p(es_df[config.TARGET].to_numpy(float))
    X_val, y_val = val_df.drop(columns=[config.TARGET]), np.log1p(val_df[config.TARGET].to_numpy(float))

    baseline_pipeline = build_pipeline(
        model_name,
        dict(config.DEFAULT_PARAMS[model_name]),
        use_original_price=use_original_price,
        tuning=model_name in {"xgboost", "lightgbm", "catboost"},
    )
    is_boosted = model_name in {"xgboost", "lightgbm", "catboost"}
    baseline_pipeline = fit_pipeline(
        baseline_pipeline,
        X_fit,
        y_fit,
        X_es if is_boosted else None,
        y_es if is_boosted else None,
        tuning=True,
    )
    baseline_pred = baseline_pipeline.predict(X_val)
    baseline_metrics = metrics.evaluate_inr(val_df[config.TARGET].to_numpy(float), baseline_pred)
    print("Baseline validation metrics:")
    print(json.dumps(baseline_metrics, indent=2))

    trial_path = config.ARTIFACTS_DIR / f"{model_name}_optuna_trials.csv"
    if trial_path.exists():
        cached_df = pd.read_csv(trial_path)
        complete = cached_df[cached_df.get("state", "COMPLETE") == "COMPLETE"] if "state" in cached_df else cached_df
        if len(complete) >= n_trials:
            study = _study_from_trial_frame(cached_df)
            best_params = dict(study.best_params)
            best_iteration = study.best_trial.user_attrs.get("best_iteration")
            if best_iteration:
                best_params["best_iteration"] = int(best_iteration)
            metadata = {
                "n_trials": len(study.trials),
                "best_value_log_rmse": float(study.best_value),
                "tuning_sample_rows": int(len(tuning_sample)),
                "fit_rows": int(len(fit_df)),
                "early_stop_rows": int(len(es_df)),
                "objective_rows": int(len(val_df)),
                "seed": config.SEED,
                "resumed_from_cache": True,
            }
            print(f"Resumed {len(study.trials)} cached trials from {trial_path.name}")
            print(f"Best validation log-RMSE: {study.best_value:.6f}")
            print("Best params:", json.dumps(best_params, indent=2, default=str))
            return study, best_params, baseline_metrics, metadata

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_params(trial, model_name)
        pipeline = build_pipeline(
            model_name,
            params,
            use_original_price=use_original_price,
            tuning=is_boosted,
        )
        model_iterations = None
        if is_boosted:
            pipeline = fit_pipeline(
                pipeline,
                X_fit,
                y_fit,
                X_es,
                y_es,
                tuning=True,
            )
            model_iterations = _best_iteration(pipeline.named_steps["model"])
        else:
            pipeline = fit_pipeline(pipeline, X_fit, y_fit)
        pred = pipeline.predict(X_val)
        rmse = float(np.sqrt(np.mean((pred - y_val) ** 2)))
        trial.set_user_attr("best_iteration", model_iterations)
        return rmse

    sampler = optuna.samplers.TPESampler(seed=config.SEED)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best_params = dict(study.best_params)
    best_iteration = study.best_trial.user_attrs.get("best_iteration")
    if best_iteration:
        best_params["best_iteration"] = int(best_iteration)
    trial_df = study.trials_dataframe()
    trial_df.to_csv(config.ARTIFACTS_DIR / f"{model_name}_optuna_trials.csv", index=False)
    metadata = {
        "n_trials": len(study.trials),
        "best_value_log_rmse": float(study.best_value),
        "tuning_sample_rows": int(len(tuning_sample)),
        "fit_rows": int(len(fit_df)),
        "early_stop_rows": int(len(es_df)),
        "objective_rows": int(len(val_df)),
        "seed": config.SEED,
    }
    print(f"Best validation log-RMSE: {study.best_value:.6f}")
    print("Best params:", json.dumps(best_params, indent=2, default=str))
    return study, best_params, baseline_metrics, metadata


def train_final_model(
    model_name: str,
    params: dict[str, Any],
    train: pd.DataFrame,
    *,
    use_original_price: bool = False,
) -> tuple[Pipeline, float]:
    """Refit the tuned pipeline on the full training split."""
    X_train = train.drop(columns=[config.TARGET])
    y_train = np.log1p(train[config.TARGET].to_numpy(float))
    if model_name == "random_forest":
        params["n_estimators"] = min(int(params.get("n_estimators", 150)), 150)
        params["max_depth"] = min(int(params.get("max_depth", 24)), 24)
    best_iteration = params.get("best_iteration")
    pipeline = build_pipeline(
        model_name,
        params,
        use_original_price=use_original_price,
        tuning=False,
        final_iterations=int(best_iteration) if best_iteration else None,
    )
    start = time.perf_counter()
    pipeline = fit_pipeline(pipeline, X_train, y_train)
    elapsed = time.perf_counter() - start
    return pipeline, float(elapsed)


def inference_time_ms(pipeline: Pipeline, X: pd.DataFrame, n_rows: int = 1000, repeats: int = 3) -> float:
    """Median inference latency in milliseconds per 1,000 rows."""
    sample = X.iloc[: min(n_rows, len(X))]
    pipeline.predict(sample)
    times: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        pipeline.predict(sample)
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000.0)
    return float(np.median(times))


def save_model_artifact(
    model_name: str,
    pipeline: Pipeline,
    params: dict[str, Any],
    *,
    use_original_price: bool,
    train_time_s: float,
    inference_ms_per_1k: float,
) -> Path:
    """Serialize a pipeline and its metadata."""
    config.ensure_dirs()
    path = config.model_path(model_name)
    payload = {
        "model_name": model_name,
        "pipeline": pipeline,
        "params": params,
        "use_original_price": use_original_price,
        "target_transform": "log1p",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "train_time_s": float(train_time_s),
        "inference_ms_per_1k": float(inference_ms_per_1k),
    }
    joblib.dump(payload, path, compress=3)
    with open(config.params_path(model_name), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "model_name": model_name,
                "params": jsonable(params),
                "use_original_price": use_original_price,
                "train_time_s": float(train_time_s),
                "inference_ms_per_1k": float(inference_ms_per_1k),
            },
            handle,
            indent=2,
            default=str,
        )
    return path


def append_metrics(model_name: str, payload: dict[str, Any]) -> None:
    """Merge a model's metrics into ``artifacts/metrics.json``."""
    config.ensure_dirs()
    path = config.METRICS_PATH
    data: dict[str, Any] = {}
    if path.exists():
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    data[model_name] = payload
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, default=str)


def load_metrics() -> dict[str, Any]:
    """Load the shared metrics file."""
    if not config.METRICS_PATH.exists():
        return {}
    with open(config.METRICS_PATH, encoding="utf-8") as handle:
        return json.load(handle)


def artifact_size_mb(model_name: str) -> float:
    """Return the serialized artifact size in megabytes."""
    path = config.model_path(model_name)
    return path.stat().st_size / (1024 * 1024) if path.exists() else 0.0


def model_feature_importance(
    pipeline: Pipeline,
    X_sample: pd.DataFrame,
    y_sample: np.ndarray,
    *,
    sample_size: int = 5_000,
) -> pd.DataFrame:
    """Return native coefficients/importances or permutation importance."""
    estimator = pipeline.named_steps["model"]
    if hasattr(estimator, "coef_"):
        names = features.transformed_feature_names(pipeline)
        values = np.ravel(estimator.coef_)
        return (
            pd.DataFrame({"feature": names, "importance": values})
            .assign(abs_importance=lambda d: d["importance"].abs())
            .sort_values("abs_importance", ascending=False)
            .reset_index(drop=True)
        )
    if hasattr(estimator, "feature_importances_"):
        names = features.transformed_feature_names(pipeline)
        values = np.ravel(estimator.feature_importances_)
        if len(names) != len(values):
            names = [f"f{i}" for i in range(len(values))]
        return (
            pd.DataFrame({"feature": names, "importance": values})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    from sklearn.inspection import permutation_importance

    sample = X_sample.iloc[:sample_size]
    y_log = np.log1p(y_sample[:sample_size])
    result = permutation_importance(
        pipeline,
        sample,
        y_log,
        n_repeats=5,
        random_state=config.SEED,
        scoring="neg_root_mean_squared_error",
        n_jobs=4,
    )
    return (
        pd.DataFrame(
            {
                "feature": sample.columns,
                "importance": result.importances_mean,
                "std": result.importances_std,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
