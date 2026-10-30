"""Fit per-model bias-correction factors on a labeled reference dataset.

By default the canonical validation split is used. To calibrate against real
labeled data, pass a CSV with the raw input columns and ``Resale_Price_INR``:

    python -m scripts.calibrate_models --input path/to/labeled.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import bias, config, data, metrics, predict  # noqa: E402


def _load_reference(input_path: str | None, split: str) -> tuple[pd.DataFrame, str]:
    if input_path:
        return data.load_raw_data(input_path), input_path
    df = data.load_raw_data()
    _, val, test = data.split_frames(df)
    frame = val if split == "val" else test
    return frame, f"canonical {split} split"


def _prediction_frame(reference: pd.DataFrame) -> pd.DataFrame:
    y_true = reference[config.TARGET].to_numpy(float)
    frame = pd.DataFrame({"y_true_inr": y_true})
    for name in predict.available_models():
        frame[name] = predict.predict_log(name, reference)
    return frame


def _print_factors(payload: dict) -> None:
    rows = []
    for stats in payload["metrics"].values():
        rows.append(
            {
                "Model": stats["model_label"],
                "Factor": f"{stats['factor']:.4f}",
                "Signed before %": f"{stats['bias_before']['mean_signed_pct']:+.2f}",
                "Signed after %": f"{stats['bias_after']['mean_signed_pct']:+.2f}",
                "MAPE before": f"{stats['before']['mape']:.2f}%",
                "MAPE after": f"{stats['after']['mape']:.2f}%",
            }
        )
    print(pd.DataFrame(rows).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="Optional labeled CSV used to fit calibration factors.")
    parser.add_argument("--eval-input", help="Optional separate labeled CSV to evaluate the factors.")
    parser.add_argument("--split", choices=["test", "val"], default="val")
    parser.add_argument(
        "--method",
        choices=["log_mean", "ratio_mean", "ratio_median", "total"],
        default="log_mean",
    )
    parser.add_argument("--output", type=Path, default=config.CALIBRATION_PATH)
    args = parser.parse_args()

    reference, source = _load_reference(args.input, args.split)
    payload = bias.fit_calibration(
        _prediction_frame(reference),
        method=args.method,
        source=source,
    )
    bias.write_json(payload, args.output)
    _print_factors(payload)
    print(f"\nWrote {args.output}")

    if args.eval_input:
        eval_frame, eval_source = _load_reference(args.eval_input, args.split)
        pred_frame = _prediction_frame(eval_frame)
        y_true = pred_frame["y_true_inr"].to_numpy(float)
        rows = []
        for name, factor in payload["factors"].items():
            if name not in pred_frame.columns:
                continue
            pred_log = pred_frame[name].to_numpy(float)
            corrected = bias.apply_factor(pred_log, factor)
            before = metrics.evaluate_inr(y_true, pred_log)
            after = metrics.evaluate_inr(y_true, corrected)
            rows.append(
                {
                    "Model": config.MODEL_LABELS.get(name, name),
                    "Factor": f"{factor:.4f}",
                    "MAPE before": f"{before['mape']:.2f}%",
                    "MAPE after": f"{after['mape']:.2f}%",
                    "Mean signed before %": f"{metrics.signed_bias(y_true, pred_log)['mean_signed_pct']:+.2f}",
                    "Mean signed after %": f"{metrics.signed_bias(y_true, corrected)['mean_signed_pct']:+.2f}",
                }
            )
        print(f"\nEvaluation on {eval_source}:")
        print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
