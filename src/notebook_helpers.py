"""High-level helpers shared by the six per-model notebooks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import learning_curve, train_test_split

from . import config, features, metrics, plots, training

FIGURES_DIR = config.ARTIFACTS_DIR / "figures"


def _save_and_show(fig: plt.Figure, name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{name}.png", dpi=110, bbox_inches="tight")
    plt.show()


def load_and_split() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load, validate, split, and create the tuning subsample."""
    df = data_load()
    data_validate(df)
    train, val, test = split(df)
    tuning = tuning_sample(train)
    return df, train, val, test, tuning


def data_load() -> pd.DataFrame:
    from . import data

    return data.load_raw_data()


def data_validate(df: pd.DataFrame) -> dict[str, Any]:
    from . import data

    return data.validate_data(df)


def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from . import data

    return data.split_frames(df)


def tuning_sample(train: pd.DataFrame) -> pd.DataFrame:
    from . import data

    return data.get_tuning_sample(train)


def predict_eval(pipeline: Any, val: pd.DataFrame, test: pd.DataFrame) -> dict[str, Any]:
    """Predict validation/test rows and compute metrics plus timings."""
    X_val = val.drop(columns=[config.TARGET])
    X_test = test.drop(columns=[config.TARGET])
    val_log = np.asarray(pipeline.predict(X_val), dtype=float)
    test_log = np.asarray(pipeline.predict(X_test), dtype=float)
    return {
        "val_log": val_log,
        "test_log": test_log,
        "val_metrics": metrics.evaluate_inr(val[config.TARGET].to_numpy(float), val_log),
        "test_metrics": metrics.evaluate_inr(test[config.TARGET].to_numpy(float), test_log),
        "inference_ms_per_1k": training.inference_time_ms(pipeline, X_test),
    }


def metrics_frame(val_metrics: dict[str, float], test_metrics: dict[str, float]) -> pd.DataFrame:
    """Build a readable validation/test metrics table."""
    rows = []
    for key, label in [
        ("r2_log", "R² (log)"),
        ("rmse_log", "RMSE (log)"),
        ("mae_log", "MAE (log)"),
        ("mae_inr", "MAE (INR)"),
        ("rmse_inr", "RMSE (INR)"),
        ("mape", "MAPE"),
        ("median_ape", "Median APE"),
    ]:
        def fmt(value: float) -> str:
            if key in {"mae_inr", "rmse_inr"}:
                return metrics.format_inr(value)
            if key in {"mape", "median_ape"}:
                return f"{value:.2f}%"
            return f"{value:.5f}"

        rows.append({"Metric": label, "Validation": fmt(val_metrics[key]), "Test": fmt(test_metrics[key])})
    return pd.DataFrame(rows).set_index("Metric")


def _transform_steps(pipeline: Any, X: pd.DataFrame) -> Any:
    transformed: Any = X
    for _, step in list(pipeline.steps)[:-1]:
        transformed = step.transform(transformed)
    return transformed


def _feature_frame(transformed: Any) -> pd.DataFrame:
    if isinstance(transformed, pd.DataFrame):
        frame = transformed.copy()
        for col in frame.select_dtypes(include=["object", "category"]).columns:
            frame[col] = frame[col].astype("category").cat.codes
        return frame
    return pd.DataFrame(np.asarray(transformed, dtype=float))


def _feature_names(pipeline: Any, transformed: Any) -> list[str]:
    if isinstance(transformed, pd.DataFrame):
        return list(transformed.columns)
    return features.transformed_feature_names(pipeline)


def shap_values_for_model(
    model_name: str,
    pipeline: Any,
    X_sample: pd.DataFrame,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Compute SHAP values for a tree model on a small raw-row sample."""
    import shap

    if model_name == "catboost":
        from catboost import Pool

        transformed = _transform_steps(pipeline, X_sample)
        cat_indices = list(range(len(transformed.columns) - len(config.CATEGORICAL_FEATURES), len(transformed.columns)))
        pool = Pool(transformed, cat_features=cat_indices)
        values = pipeline.named_steps["model"].get_feature_importance(pool, type="ShapValues")
        values = np.asarray(values)[:, :-1]
        return values, _feature_frame(transformed)

    transformed = _transform_steps(pipeline, X_sample)
    if model_name == "lightgbm":
        frame = _feature_frame(transformed)
        matrix = frame.to_numpy(dtype=float)
    else:
        matrix = np.asarray(transformed, dtype=float)
        frame = pd.DataFrame(matrix, columns=_feature_names(pipeline, transformed))

    if model_name == "random_forest":
        background = shap.sample(matrix, min(50, len(matrix)), random_state=config.SEED)
        explainer = shap.PermutationExplainer(
            pipeline.named_steps["model"].predict,
            background,
        )
        values = explainer(matrix, max_evals=2 * matrix.shape[1] + 1).values
        return np.asarray(values), frame

    explainer = shap.TreeExplainer(
        pipeline.named_steps["model"],
        feature_perturbation="tree_path_dependent",
    )
    values = explainer.shap_values(matrix, check_additivity=False)
    if isinstance(values, list):
        values = values[0]
    return np.asarray(values), frame


def partial_dependence_curves(pipeline: Any, X_sample: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Compute model-agnostic PDP-style curves on raw input rows."""
    base = X_sample.sample(min(500, len(X_sample)), random_state=config.SEED).copy()

    age_rows = []
    for age in range(0, 18):
        frame = base.copy()
        frame["Manufacturing_Year"] = config.REF_YEAR - age
        frame["Year_Purchased"] = np.maximum(frame["Year_Purchased"], frame["Manufacturing_Year"])
        pred = np.expm1(pipeline.predict(frame))
        age_rows.append({"x": age, "y": float(np.mean(pred))})

    km_values = np.unique(
        np.clip(
            np.linspace(1_000, max(200_000, float(base["Kilometres_Driven"].quantile(0.99))), 12),
            300,
            400_000,
        ).round()
    )
    km_rows = []
    for km in km_values:
        frame = base.copy()
        frame["Kilometres_Driven"] = int(km)
        pred = np.expm1(pipeline.predict(frame))
        km_rows.append({"x": int(km), "y": float(np.mean(pred))})

    condition_rows = []
    for condition in config.CONDITION_LEVELS:
        frame = base.copy()
        frame["Current_Condition"] = condition
        pred = np.expm1(pipeline.predict(frame))
        condition_rows.append({"x": condition, "y": float(np.mean(pred))})

    return {
        "Age": pd.DataFrame(age_rows),
        "Kilometres_Driven": pd.DataFrame(km_rows),
        "Current_Condition": pd.DataFrame(condition_rows),
    }


def learning_curve_scores(
    model_name: str,
    best_params: dict[str, Any],
    train: pd.DataFrame,
    sample_size: int = 12_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a small RMSE learning curve without cloning the pipeline.

    Fitting fresh pipelines avoids sklearn's clone check, which CatBoost's
    ``cat_features`` parameter does not satisfy.
    """
    sample = train.sample(min(sample_size, len(train)), random_state=config.SEED)
    X = sample.drop(columns=[config.TARGET])
    y = np.log1p(sample[config.TARGET].to_numpy(float))
    best_iteration = best_params.get("best_iteration")
    sizes, train_scores, val_scores = [], [], []
    for frac in (0.5, 0.75, 1.0):
        if frac < 1.0:
            X_sub, _, y_sub, _ = train_test_split(
                X, y, train_size=frac, random_state=config.SEED
            )
        else:
            X_sub, y_sub = X, y
        X_fit, X_hold, y_fit, y_hold = train_test_split(
            X_sub, y_sub, test_size=0.2, random_state=config.SEED
        )
        pipeline = training.build_pipeline(
            model_name,
            best_params,
            final_iterations=int(best_iteration) if best_iteration else None,
        )
        pipeline = training.fit_pipeline(pipeline, X_fit, y_fit)
        train_rmse = float(np.sqrt(np.mean((pipeline.predict(X_fit) - y_fit) ** 2)))
        val_rmse = float(np.sqrt(np.mean((pipeline.predict(X_hold) - y_hold) ** 2)))
        sizes.append(len(X_sub))
        train_scores.append(train_rmse)
        val_scores.append(val_rmse)
    return np.asarray(sizes), np.asarray(train_scores), np.asarray(val_scores)


def run_standard_plots(
    model_name: str,
    pipeline: Any,
    best_params: dict[str, Any],
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    test_log: np.ndarray,
) -> None:
    """Generate all required diagnostic plots for a model notebook."""
    plots.set_style()
    label = config.MODEL_LABELS[model_name]
    y_test_inr = test[config.TARGET].to_numpy(float)
    residual = np.log1p(y_test_inr) - test_log

    _save_and_show(
        plots.plot_pred_vs_actual(y_test_inr, test_log, f"{label}: predicted vs actual (test)"),
        f"{model_name}_pred_vs_actual",
    )
    _save_and_show(
        plots.plot_residuals(test_log, residual, f"{label}: residuals vs predictions"),
        f"{model_name}_residuals",
    )
    _save_and_show(
        plots.plot_residual_hist_qq(residual, label),
        f"{model_name}_residual_qq",
    )
    _save_and_show(
        plots.plot_error_segments(test, y_test_inr, test_log, f"{label}: error by segment"),
        f"{model_name}_error_segments",
    )

    importance = training.model_feature_importance(
        pipeline,
        val.drop(columns=[config.TARGET]),
        val[config.TARGET].to_numpy(float),
        sample_size=5_000,
    )
    _save_and_show(
        plots.plot_feature_importance(importance, f"{label}: feature importance"),
        f"{model_name}_importance",
    )

    if model_name in config.TREE_MODELS:
        try:
            X_shap = test.drop(columns=[config.TARGET]).sample(
                min(200 if model_name == "random_forest" else 500, len(test)),
                random_state=config.SEED,
            )
            shap_values, shap_frame = shap_values_for_model(model_name, pipeline, X_shap)
            _save_and_show(
                plots.plot_shap_summary(shap_values, shap_frame, f"{label}: SHAP summary"),
                f"{model_name}_shap",
            )
        except Exception as exc:  # pragma: no cover - optional diagnostic
            print(f"SHAP summary unavailable for {label}: {exc}")

    pdp = partial_dependence_curves(pipeline, test.drop(columns=[config.TARGET]))
    _save_and_show(
        plots.plot_partial_dependence(pdp, f"{label}: partial dependence"),
        f"{model_name}_pdp",
    )

    if model_name == "mlp":
        loss = getattr(pipeline.named_steps["model"], "loss_curve_", None)
        if loss is not None:
            _save_and_show(
                plots.plot_loss_curve(loss, f"{label}: training loss curve"),
                f"{model_name}_loss_curve",
            )
    else:
        sizes, train_scores, val_scores = learning_curve_scores(model_name, best_params, train)
        _save_and_show(
            plots.plot_learning_curve(sizes, train_scores, val_scores, f"{label}: learning curve"),
            f"{model_name}_learning_curve",
        )


def sanity_report(
    model_name: str,
    pipeline: Any,
    test: pd.DataFrame,
) -> pd.DataFrame:
    """Run monotonicity, EV, and price-cap sanity checks."""
    from scipy.stats import spearmanr

    base_candidates = test[test["Model"] == "Swift"]
    base = (base_candidates if len(base_candidates) else test).iloc[[0]].copy()

    age_frame = base.copy()
    age_prices = []
    for age in range(0, 18):
        frame = base.copy()
        frame["Manufacturing_Year"] = config.REF_YEAR - age
        frame["Year_Purchased"] = np.maximum(frame["Year_Purchased"], frame["Manufacturing_Year"])
        age_prices.append(float(np.expm1(pipeline.predict(frame))[0]))
    age_prices_arr = np.asarray(age_prices)
    age_steps = np.diff(age_prices_arr)
    age_rho = spearmanr(np.arange(18), age_prices_arr).statistic

    km_prices = []
    for km in np.linspace(5_000, 300_000, 30):
        frame = base.copy()
        frame["Kilometres_Driven"] = int(km)
        km_prices.append(float(np.expm1(pipeline.predict(frame))[0]))
    km_prices_arr = np.asarray(km_prices)
    km_steps = np.diff(km_prices_arr)
    km_rho = spearmanr(np.arange(len(km_prices_arr)), km_prices_arr).statistic

    ev = test[test["Fuel_Type"] == "Electric"]
    ev_metrics = None
    if len(ev):
        ev_log = pipeline.predict(ev.drop(columns=[config.TARGET]))
        ev_metrics = metrics.evaluate_inr(ev[config.TARGET].to_numpy(float), ev_log)

    predicted_inr = np.expm1(pipeline.predict(test.drop(columns=[config.TARGET])))
    original = test[config.OPTIONAL_PRICE_COLUMN].to_numpy(float)
    pred_below_original = float(np.mean(predicted_inr < original) * 100)
    actual_below_original = float(np.mean(test[config.TARGET].to_numpy(float) < original) * 100)

    rows = [
        {
            "Check": "Age monotonicity",
            "Result": (
                f"Spearman ρ={age_rho:.3f}; "
                f"{np.mean(age_steps <= 0) * 100:.1f}% of age steps lower price"
            ),
            "Status": "pass" if age_rho < -0.8 else "review",
        },
        {
            "Check": "Kilometres monotonicity",
            "Result": (
                f"Spearman ρ={km_rho:.3f}; "
                f"{np.mean(km_steps <= 0) * 100:.1f}% of km steps lower price"
            ),
            "Status": "pass" if km_rho < -0.6 else "review",
        },
        {
            "Check": "EV rows",
            "Result": (
                "No EV test rows"
                if ev_metrics is None
                else f"n={len(ev):,}; MAPE={ev_metrics['mape']:.2f}%; RMSE(log)={ev_metrics['rmse_log']:.4f}"
            ),
            "Status": "pass" if ev_metrics is None or ev_metrics["mape"] < 10 else "review",
        },
        {
            "Check": "Predicted resale < original",
            "Result": f"{pred_below_original:.2f}% predicted below original; actual {actual_below_original:.2f}%",
            "Status": "pass" if pred_below_original >= 99.0 else "review",
        },
    ]
    frame = pd.DataFrame(rows)
    print(frame.to_string(index=False))
    return frame


def save_and_report(
    model_name: str,
    pipeline: Any,
    best_params: dict[str, Any],
    baseline_metrics: dict[str, float],
    tuning_meta: dict[str, Any],
    evaluation: dict[str, Any],
    train_time_s: float,
) -> dict[str, Any]:
    """Save the artifact, append metrics, and return the metrics payload."""
    inference_ms = float(evaluation["inference_ms_per_1k"])
    training.save_model_artifact(
        model_name,
        pipeline,
        best_params,
        use_original_price=config.USE_ORIGINAL_PRICE,
        train_time_s=train_time_s,
        inference_ms_per_1k=inference_ms,
    )
    payload = {
        "model_label": config.MODEL_LABELS[model_name],
        "use_original_price": config.USE_ORIGINAL_PRICE,
        "val": evaluation["val_metrics"],
        "test": evaluation["test_metrics"],
        "baseline_val": baseline_metrics,
        "tuning": tuning_meta,
        "params": training.jsonable(best_params),
        "train_time_s": float(train_time_s),
        "inference_ms_per_1k": inference_ms,
        "size_mb": training.artifact_size_mb(model_name),
        "noise_floor_note": "Generator noise is ~4.5% log-normal; MAPE near 4-6% is expected.",
    }
    training.append_metrics(model_name, payload)
    print("Saved artifact and appended metrics:")
    print(json.dumps({k: v for k, v in payload.items() if k not in {"params"}}, indent=2, default=str))
    return payload
