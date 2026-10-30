"""Report signed prediction bias for every saved model.

Run on the canonical test split:

    python -m scripts.bias_report

Run on an external labeled CSV (must contain the raw input columns and the target):

    python -m scripts.bias_report --input path/to/labeled.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import bias, config, data, predict  # noqa: E402


def _load_reference(input_path: str | None, split: str) -> tuple[pd.DataFrame, str]:
    if input_path:
        frame = data.load_raw_data(input_path)
        return frame, input_path
    df = data.load_raw_data()
    train, val, test = data.split_frames(df)
    frame = test if split == "test" else val
    return frame, f"canonical {split} split"


def _prediction_frame(reference: pd.DataFrame) -> pd.DataFrame:
    y_true = reference[config.TARGET].to_numpy(float)
    frame = pd.DataFrame({"y_true_inr": y_true, "y_true_log": np.log1p(y_true)})
    for name in predict.available_models():
        frame[name] = predict.predict_log(name, reference)
    return frame


def _print_summary(report: dict) -> None:
    rows = []
    for name, stats in report["global"].items():
        rows.append(
            {
                "Model": stats["model_label"],
                "Rows": stats["n"],
                "Mean signed %": f"{stats['mean_signed_pct']:+.2f}",
                "Median signed %": f"{stats['median_signed_pct']:+.2f}",
                "Pred/Actual": f"{stats['mean_ratio']:.4f}",
                "Under %": f"{stats['underprediction_rate']:.1f}",
                "Factor": f"{stats['calibration_factor_log']:.4f}",
            }
        )
    print(pd.DataFrame(rows).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="Optional labeled CSV with raw inputs and the target column.")
    parser.add_argument("--split", choices=["test", "val"], default="test")
    parser.add_argument("--min-rows", type=int, default=30)
    parser.add_argument("--output", type=Path, default=config.BIAS_REPORT_PATH)
    args = parser.parse_args()

    reference, source = _load_reference(args.input, args.split)
    pred_frame = _prediction_frame(reference)
    report = bias.build_bias_report(
        pred_frame,
        reference,
        source=source,
        min_rows=args.min_rows,
    )
    report["min_segment_rows"] = args.min_rows
    bias.write_json(report, args.output)
    _print_summary(report)
    print("\nNegative signed percentages mean under-prediction.")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
