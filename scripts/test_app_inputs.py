"""Exercise five varied dashboard inputs through src.predict and report all models."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config, data, metrics, predict  # noqa: E402


def _row(df: pd.DataFrame, mask: pd.Series, sort_by: str | None = None) -> dict:
    subset = df[mask]
    if sort_by and sort_by in subset.columns:
        subset = subset.sort_values(sort_by)
    if subset.empty:
        raise ValueError("No matching row for scenario")
    row = subset.iloc[0][config.RAW_INPUT_COLUMNS].to_dict()
    row["Original_Purchase_Price_INR"] = 0.0
    return row


def scenarios(df: pd.DataFrame) -> dict[str, dict]:
    age = config.REF_YEAR - df["Manufacturing_Year"]
    ev_mask = (df["Fuel_Type"] == "Electric") & (df["Model"].str.contains("Nexon", case=False, na=False))
    if not ev_mask.any():
        ev_mask = df["Fuel_Type"] == "Electric"
    luxury_mask = (df["Manufacturer"] == "BMW") & (df["Model"] == "X5") & (df["Manufacturing_Year"] >= 2021)
    old_delhi_mask = (
        (df["Fuel_Type"] == "Diesel")
        & (df["Location"] == "Delhi")
        & ((config.REF_YEAR - df["Manufacturing_Year"]) >= 11)
        & (df["Manufacturer"].isin(["Toyota", "Mahindra", "Hyundai"]))
    )
    high_km_mask = (df["Model"] == "Swift") & (df["Kilometres_Driven"] >= 250_000)
    new_mask = (
        (df["Model"].isin(["Creta", "Seltos", "Nexon"]))
        & (df["Manufacturing_Year"] >= 2025)
        & (df["Number_of_Previous_Owners"] == 0)
        & (df["Kilometres_Driven"] <= 20_000)
    )
    return {
        "EV": _row(df, ev_mask, "Manufacturing_Year"),
        "Old Delhi diesel": _row(df, old_delhi_mask, "Manufacturing_Year"),
        "Luxury car": _row(df, luxury_mask, "Manufacturing_Year"),
        "High-km car": _row(df, high_km_mask, "Kilometres_Driven"),
        "First-owner nearly new": _row(df, new_mask, "Manufacturing_Year"),
    }


def main() -> None:
    df = data.load_raw_data()
    available = predict.available_models()
    results: dict[str, dict[str, float]] = {}
    for label, row in scenarios(df).items():
        frame = pd.DataFrame([row])
        predictions = {name: float(predict.predict_inr(name, frame)[0]) for name in available}
        if len(available) == len(config.MODEL_NAMES):
            predictions["ensemble_all"] = float(predict.predict_ensemble(frame, available)[0])
        results[label] = predictions
        print(f"\n=== {label} ===")
        print(
            f"{row['Manufacturer']} {row['Model']} | {row['Fuel_Type']} {row['Transmission']} | "
            f"MY {row['Manufacturing_Year']} | {row['Kilometres_Driven']:,} km | "
            f"{row['Location']} | {row['Current_Condition']} | {row['Number_of_Previous_Owners']} owners"
        )
        for name, value in predictions.items():
            print(f"  {name:15s} {metrics.format_inr(value)}")

    output = config.ARTIFACTS_DIR / "app_scenarios.json"
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(f"\nWrote {output}")


if __name__ == "__main__":
    main()
