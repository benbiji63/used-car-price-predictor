"""Shared plotting helpers with a consistent visual style."""

from __future__ import annotations

import math
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from scipy import stats

from . import config, metrics

PALETTE = "viridis"
MODEL_COLORS = {
    "Ridge": "#4C72B0",
    "Random Forest": "#DD8452",
    "XGBoost": "#55A868",
    "LightGBM": "#C44E52",
    "CatBoost": "#8172B3",
    "MLP": "#937860",
    "Ensemble (all)": "#DA8BC3",
    "Ensemble (top-3)": "#8C8C8C",
}


def set_style() -> None:
    """Apply a consistent theme for notebooks and saved figures."""
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "figure.figsize": (10, 6),
            "figure.dpi": 110,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.family": "DejaVu Sans",
            "axes.unicode_minus": False,
        }
    )


def show(fig: plt.Figure | None = None) -> None:
    """Show a figure with tight layout, closing it afterwards."""
    if fig is not None:
        fig.tight_layout()
    plt.show()


def _inr_axis(value: float, _position: int) -> str:
    return metrics.compact_inr(value)


def plot_optuna_history(study: Any, title: str) -> plt.Figure:
    """Plot trial values and the running best objective."""
    trials = [t for t in study.trials if t.value is not None]
    values = np.asarray([t.value for t in trials], dtype=float)
    best = np.minimum.accumulate(values)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(np.arange(1, len(values) + 1), values, s=18, alpha=0.55, label="Trial")
    ax.step(np.arange(1, len(values) + 1), best, where="post", color="#C44E52", label="Best so far")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Validation RMSE (log)")
    ax.set_title(title)
    ax.legend()
    return fig


def plot_pred_vs_actual(y_true_inr: np.ndarray, pred_log: np.ndarray, title: str) -> plt.Figure:
    """Hexbin predicted-vs-actual plot in log space."""
    y_log = np.log1p(np.asarray(y_true_inr, dtype=float))
    pred_log = np.asarray(pred_log, dtype=float)
    lo = float(min(y_log.min(), pred_log.min()))
    hi = float(max(y_log.max(), pred_log.max()))
    fig, ax = plt.subplots(figsize=(8, 7))
    hb = ax.hexbin(y_log, pred_log, gridsize=65, bins="log", cmap=PALETTE, mincnt=1)
    ax.plot([lo, hi], [lo, hi], color="#C44E52", linewidth=1.5, label="y = x")
    ax.set_xlabel("Actual log1p(price)")
    ax.set_ylabel("Predicted log1p(price)")
    ax.set_title(title)
    ax.legend(loc="upper left")
    fig.colorbar(hb, ax=ax, label="log10(count)")
    return fig


def plot_residuals(pred_log: np.ndarray, residual_log: np.ndarray, title: str) -> plt.Figure:
    """Hexbin residuals versus predictions."""
    fig, ax = plt.subplots(figsize=(9, 5))
    hb = ax.hexbin(pred_log, residual_log, gridsize=65, bins="log", cmap=PALETTE, mincnt=1)
    ax.axhline(0.0, color="#C44E52", linewidth=1.2)
    ax.set_xlabel("Predicted log1p(price)")
    ax.set_ylabel("Residual (actual − predicted, log)")
    ax.set_title(title)
    fig.colorbar(hb, ax=ax, label="log10(count)")
    return fig


def plot_residual_hist_qq(residual_log: np.ndarray, title: str) -> plt.Figure:
    """Residual histogram and normal QQ plot."""
    residual_log = np.asarray(residual_log, dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.histplot(residual_log, bins=80, kde=True, ax=axes[0], color="#4C72B0")
    axes[0].set_title(f"{title} — residual distribution")
    axes[0].set_xlabel("Residual (log)")
    stats.probplot(residual_log, dist="norm", plot=axes[1])
    axes[1].set_title("Normal QQ plot")
    axes[1].set_xlabel("Theoretical quantiles")
    axes[1].set_ylabel("Ordered residuals")
    return fig


def _ape_series(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1.0)) * 100.0


def plot_error_segments(
    test_df: pd.DataFrame,
    y_true_inr: np.ndarray,
    pred_log: np.ndarray,
    title: str,
) -> plt.Figure:
    """MAPE by price band, age, fuel type, and Delhi restriction."""
    pred_inr = np.expm1(np.asarray(pred_log, dtype=float))
    frame = pd.DataFrame(
        {
            "actual": np.asarray(y_true_inr, dtype=float),
            "pred": np.maximum(pred_inr, 0.0),
            "age": test_df["Age"].to_numpy() if "Age" in test_df else config.REF_YEAR - test_df["Manufacturing_Year"].to_numpy(),
            "fuel": test_df["Fuel_Type"].to_numpy(),
            "delhi": np.where(
                test_df["Location"].isin(config.NCR_CITIES)
                & (
                    ((test_df["Fuel_Type"] == "Diesel") & ((config.REF_YEAR - test_df["Manufacturing_Year"]) >= 10))
                    | ((test_df["Fuel_Type"] == "Petrol") & ((config.REF_YEAR - test_df["Manufacturing_Year"]) >= 15))
                ),
                "Restricted NCR",
                "Not restricted",
            ),
        }
    )
    frame["ape"] = _ape_series(frame["actual"], frame["pred"])
    frame["price_band"] = pd.qcut(frame["actual"], 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    sns.barplot(data=frame, x="price_band", y="ape", ax=axes[0, 0], color="#4C72B0", errorbar=None)
    axes[0, 0].set_title("MAPE by actual price quintile")
    axes[0, 0].set_xlabel("Price band")
    axes[0, 0].set_ylabel("MAPE (%)")

    age_bands = pd.cut(frame["age"], bins=[-1, 2, 5, 8, 11, 14, 20], labels=["0-2", "3-5", "6-8", "9-11", "12-14", "15+"])
    sns.barplot(x=age_bands, y=frame["ape"], ax=axes[0, 1], color="#55A868", errorbar=None)
    axes[0, 1].set_title("MAPE by vehicle age (years)")
    axes[0, 1].set_xlabel("Age band")
    axes[0, 1].set_ylabel("MAPE (%)")

    sns.barplot(data=frame, x="fuel", y="ape", ax=axes[1, 0], color="#DD8452", errorbar=None)
    axes[1, 0].set_title("MAPE by fuel type")
    axes[1, 0].set_xlabel("Fuel type")
    axes[1, 0].set_ylabel("MAPE (%)")
    axes[1, 0].tick_params(axis="x", rotation=15)

    sns.barplot(data=frame, x="delhi", y="ape", ax=axes[1, 1], color="#C44E52", errorbar=None)
    axes[1, 1].set_title("MAPE by Delhi-NCR restriction")
    axes[1, 1].set_xlabel("Restriction status")
    axes[1, 1].set_ylabel("MAPE (%)")
    fig.suptitle(title, fontsize=14, fontweight="bold")
    return fig


def plot_feature_importance(importance: pd.DataFrame, title: str, top_n: int = 25) -> plt.Figure:
    """Horizontal bar chart of feature importance."""
    frame = importance.head(top_n).iloc[::-1]
    value_col = "importance" if "importance" in frame.columns else "abs_importance"
    fig, ax = plt.subplots(figsize=(9, max(4, 0.32 * len(frame) + 2)))
    ax.barh(frame["feature"], frame[value_col], color="#4C72B0")
    ax.set_xlabel("Importance")
    ax.set_title(title)
    return fig


def plot_shap_summary(
    shap_values: np.ndarray,
    features: pd.DataFrame,
    title: str,
    max_display: int = 20,
) -> plt.Figure:
    """SHAP beeswarm summary plot, with a bar fallback on failure."""
    import shap

    fig = plt.figure(figsize=(10, 7))
    try:
        shap.summary_plot(
            shap_values,
            features,
            max_display=max_display,
            show=False,
            plot_type=None,
        )
        fig = plt.gcf()
        fig.set_size_inches(10, 7)
        fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    except Exception as exc:  # pragma: no cover - visualization fallback
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(10, 6))
        mean_abs = np.abs(np.asarray(shap_values)).mean(axis=0)
        order = np.argsort(mean_abs)[-max_display:]
        ax.barh(np.asarray(features.columns)[order], mean_abs[order], color="#4C72B0")
        ax.set_title(f"{title} — mean |SHAP| (summary fallback: {exc})")
        ax.set_xlabel("Mean |SHAP value|")
    return fig


def plot_partial_dependence(curves: dict[str, pd.DataFrame], title: str) -> plt.Figure:
    """Plot custom partial-dependence curves for three features."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, (name, frame) in zip(axes, curves.items()):
        ax.plot(frame["x"], frame["y"], marker="o", color="#4C72B0")
        ax.set_title(name)
        ax.set_xlabel(name)
        ax.set_ylabel("Mean predicted price")
        ax.yaxis.set_major_formatter(FuncFormatter(_inr_axis))
        if name == "Current_Condition":
            ax.tick_params(axis="x", rotation=20)
    fig.suptitle(title, fontsize=14, fontweight="bold")
    return fig


def plot_learning_curve(
    train_sizes: np.ndarray,
    train_scores: np.ndarray,
    validation_scores: np.ndarray,
    title: str,
) -> plt.Figure:
    """Plot RMSE learning curves from sklearn's learning_curve output."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(train_sizes, train_scores, marker="o", label="Train RMSE", color="#4C72B0")
    ax.plot(train_sizes, validation_scores, marker="o", label="Validation RMSE", color="#C44E52")
    ax.set_xlabel("Training rows")
    ax.set_ylabel("RMSE (log)")
    ax.set_title(title)
    ax.legend()
    return fig


def plot_loss_curve(loss_curve: Iterable[float], title: str) -> plt.Figure:
    """Plot the MLP training loss curve."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(np.arange(1, len(loss_curve) + 1), list(loss_curve), color="#4C72B0")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training loss")
    ax.set_title(title)
    return fig


def plot_metric_bars(leaderboard: pd.DataFrame) -> plt.Figure:
    """Grouped bars for RMSE(log), MAPE, and median APE."""
    metrics_to_plot = ["rmse_log", "mape", "median_ape"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, metric in zip(axes, metrics_to_plot):
        order = leaderboard.sort_values(metric).reset_index(drop=True)
        colors = [MODEL_COLORS.get(name, "#4C72B0") for name in order["Model"]]
        ax.bar(order["Model"], order[metric], color=colors)
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=30)
        ax.set_ylabel(metric)
    fig.suptitle("Model comparison on the untouched test set", fontsize=14, fontweight="bold")
    return fig


def plot_ape_box(ape_frame: pd.DataFrame, title: str) -> plt.Figure:
    """Box plot of absolute percentage errors by model."""
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.boxplot(data=ape_frame, x="Model", y="APE", ax=ax, showfliers=False, palette=PALETTE)
    ax.set_title(title)
    ax.set_ylabel("Absolute percentage error (%)")
    ax.tick_params(axis="x", rotation=25)
    return fig


def plot_small_multiples(predictions: dict[str, np.ndarray], y_true_inr: np.ndarray, title: str) -> plt.Figure:
    """Predicted-vs-actual hexbin small multiples for several models."""
    names = list(predictions)
    n_cols = 3
    n_rows = math.ceil(len(names) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4.6 * n_rows))
    axes = np.atleast_1d(axes).ravel()
    y_log = np.log1p(np.asarray(y_true_inr, dtype=float))
    lo = float(y_log.min())
    hi = float(y_log.max())
    for ax, name in zip(axes, names):
        pred = np.asarray(predictions[name], dtype=float)
        ax.hexbin(y_log, pred, gridsize=45, bins="log", cmap=PALETTE, mincnt=1)
        ax.plot([lo, hi], [lo, hi], color="#C44E52", linewidth=1.0)
        ax.set_title(name)
        ax.set_xlabel("Actual (log)")
        ax.set_ylabel("Predicted (log)")
    for ax in axes[len(names):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=14, fontweight="bold")
    return fig


def plot_error_by_segments_all(
    frame: pd.DataFrame,
    title: str = "Absolute percentage error by segment for all models",
) -> plt.Figure:
    """Compare model errors across price band, age, and fuel."""
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for ax, column in zip(axes, ["price_band", "age_band", "fuel"]):
        sns.lineplot(data=frame, x=column, y="APE", hue="Model", marker="o", ax=ax)
        ax.set_title(f"APE by {column.replace('_', ' ')}")
        ax.set_ylabel("MAPE (%)")
        ax.tick_params(axis="x", rotation=20)
        if ax.get_legend() is not None:
            ax.get_legend().remove()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle(title, fontsize=14, fontweight="bold")
    return fig


def plot_train_time_vs_error(leaderboard: pd.DataFrame) -> plt.Figure:
    """Scatter of training time versus test RMSE."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for _, row in leaderboard.iterrows():
        ax.scatter(row["train_time_s"], row["rmse_log"], s=90, color=MODEL_COLORS.get(row["Model"], "#4C72B0"))
        ax.annotate(row["Model"], (row["train_time_s"], row["rmse_log"]), xytext=(5, 5), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("Training time (seconds, log scale)")
    ax.set_ylabel("Test RMSE (log)")
    ax.set_title("Training cost versus error")
    return fig


def plot_time_size(leaderboard: pd.DataFrame) -> plt.Figure:
    """Inference latency and artifact size comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    order = leaderboard.sort_values("inference_ms_per_1k")
    axes[0].barh(order["Model"], order["inference_ms_per_1k"], color="#55A868")
    axes[0].set_xlabel("Milliseconds per 1,000 rows")
    axes[0].set_title("Inference latency")
    order = leaderboard.sort_values("size_mb")
    axes[1].barh(order["Model"], order["size_mb"], color="#DD8452")
    axes[1].set_xlabel("Artifact size (MB)")
    axes[1].set_title("Serialized model size")
    fig.suptitle("Deployment cost comparison", fontsize=14, fontweight="bold")
    return fig


def plot_residual_correlation(corr: pd.DataFrame) -> plt.Figure:
    """Heatmap of pairwise log-residual correlations."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax, square=True)
    ax.set_title("Residual correlation between models")
    return fig


def plot_generalization(degradation: pd.DataFrame, title: str) -> plt.Figure:
    """Grouped bars comparing in-distribution and held-out RMSE."""
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(degradation))
    width = 0.38
    ax.bar(x - width / 2, degradation["baseline_rmse_log"], width, label="Baseline", color="#4C72B0")
    ax.bar(x + width / 2, degradation["holdout_rmse_log"], width, label="Held out", color="#C44E52")
    ax.set_xticks(x)
    ax.set_xticklabels(degradation["Model"], rotation=25)
    ax.set_ylabel("RMSE (log)")
    ax.set_title(title)
    ax.legend()
    return fig


def plot_delhi_rule(frame: pd.DataFrame) -> plt.Figure:
    """Plot actual and predicted Delhi/non-Delhi price ratios by age."""
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.lineplot(data=frame, x="Age", y="ratio", hue="Series", marker="o", ax=ax)
    ax.set_xlabel("Diesel vehicle age (years)")
    ax.set_ylabel("Delhi-NCR / non-NCR price ratio")
    ax.set_title("Does the model learn the Delhi-NCR diesel penalty?")
    return fig


