"""Quick 20k-row smoke test for all six pipelines."""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from src import config, data, metrics, training  # noqa: E402


def main() -> None:
    config.ensure_dirs()
    df = data.load_raw_data()
    data.validate_data(df)
    train_idx, _, _ = data.make_splits(df)
    train = df.iloc[train_idx].reset_index(drop=True)
    sample = train.sample(n=20_000, random_state=config.SEED).reset_index(drop=True)
    cut = int(len(sample) * 0.8)
    fit_df, eval_df = sample.iloc[:cut], sample.iloc[cut:]
    X_fit = fit_df.drop(columns=[config.TARGET])
    y_fit = np.log1p(fit_df[config.TARGET].to_numpy(float))
    for model_name in config.MODEL_NAMES:
        start = time.perf_counter()
        pipeline = training.build_pipeline(
            model_name,
            dict(config.DEFAULT_PARAMS[model_name]),
            tuning=model_name in {"xgboost", "lightgbm", "catboost"},
        )
        pipeline = training.fit_pipeline(
            pipeline,
            X_fit,
            y_fit,
            eval_df.drop(columns=[config.TARGET]) if model_name in {"xgboost", "lightgbm", "catboost"} else None,
            np.log1p(eval_df[config.TARGET].to_numpy(float)) if model_name in {"xgboost", "lightgbm", "catboost"} else None,
            tuning=True,
        )
        pred = pipeline.predict(eval_df.drop(columns=[config.TARGET]))
        result = metrics.evaluate_inr(eval_df[config.TARGET].to_numpy(float), pred)
        print(
            f"{config.MODEL_LABELS[model_name]:15s} "
            f"rmse_log={result['rmse_log']:.4f} mape={result['mape']:.2f}% "
            f"time={time.perf_counter() - start:.1f}s"
        )
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
