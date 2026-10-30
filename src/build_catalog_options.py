"""Build the cascading dropdown catalog used by the Streamlit dashboard."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from typing import Any

import pandas as pd

from . import config, data


def _sorted_int(values: set[float]) -> list[int]:
    return [int(round(v)) for v in sorted(values)]


def build_catalog(df: pd.DataFrame) -> dict[str, Any]:
    """Build manufacturer/model/fuel/transmission/displacement options."""
    train_idx, _, _ = data.make_splits(df)
    train = df.iloc[train_idx]
    age = config.REF_YEAR - train["Manufacturing_Year"].astype(int)

    km_q99_by_age = (
        pd.DataFrame({"Age": age, "km": train["Kilometres_Driven"]})
        .groupby("Age")["km"]
        .quantile(0.99)
        .round()
        .astype(int)
        .to_dict()
    )
    km_median_by_age = (
        pd.DataFrame({"Age": age, "km": train["Kilometres_Driven"]})
        .groupby("Age")["km"]
        .median()
        .round()
        .astype(int)
        .to_dict()
    )

    grouped: dict[tuple[str, str], pd.DataFrame] = {
        key: frame for key, frame in df.groupby(["Manufacturer", "Model"], observed=True)
    }
    manufacturer_counts = Counter(df["Manufacturer"])
    manufacturers = sorted(manufacturer_counts, key=lambda x: (-manufacturer_counts[x], x))

    models_by_manufacturer: dict[str, list[str]] = defaultdict(list)
    model_info: dict[str, Any] = {}
    model_rows: list[tuple[str, str, int]] = []
    for (manufacturer, model), frame in grouped.items():
        model_rows.append((manufacturer, model, len(frame)))
        model_key = f"{manufacturer}|{model}"
        fuels = sorted(frame["Fuel_Type"].unique().tolist())
        transmissions_by_fuel: dict[str, list[str]] = {}
        displacements: dict[str, list[int]] = {}
        for fuel in fuels:
            fuel_frame = frame[frame["Fuel_Type"] == fuel]
            transmissions = sorted(fuel_frame["Transmission"].unique().tolist())
            transmissions_by_fuel[fuel] = transmissions
            for transmission in transmissions:
                subset = fuel_frame[fuel_frame["Transmission"] == transmission]
                key = f"{fuel}|{transmission}"
                displacements[key] = _sorted_int(set(subset["Engine_Displacement_cc"].astype(float)))
        model_info[model_key] = {
            "manufacturer": manufacturer,
            "model": model,
            "body_types": sorted(frame["Body_Type"].unique().tolist()),
            "fuels": fuels,
            "transmissions_by_fuel": transmissions_by_fuel,
            "displacements_by_fuel_transmission": displacements,
            "year_min": int(frame["Manufacturing_Year"].min()),
            "year_max": int(frame["Manufacturing_Year"].max()),
            "trim_levels": [
                trim for trim in config.TRIM_LEVELS
                if trim in set(frame["Variant_Trim"].unique())
            ],
            "count": int(len(frame)),
        }

    for manufacturer, model, _ in sorted(model_rows, key=lambda x: (-x[2], x[0], x[1])):
        models_by_manufacturer[manufacturer].append(model)

    locations = sorted(df["Location"].unique().tolist())
    default_key = "Maruti Suzuki|Swift"
    if default_key not in model_info:
        default_key = next(iter(model_info))
    default_model = model_info[default_key]
    default_fuel = "Petrol" if "Petrol" in default_model["fuels"] else default_model["fuels"][0]
    default_transmissions = default_model["transmissions_by_fuel"][default_fuel]
    default_transmission = "Manual" if "Manual" in default_transmissions else default_transmissions[0]
    default_cc_options = default_model["displacements_by_fuel_transmission"][f"{default_fuel}|{default_transmission}"]
    default_cc = 1197 if 1197 in default_cc_options else default_cc_options[0]
    default_manufacture_year = max(default_model["year_min"], min(2021, default_model["year_max"]))

    return {
        "catalog_version": 1,
        "reference_year": config.REF_YEAR,
        "manufacturers": manufacturers,
        "models_by_manufacturer": {m: models_by_manufacturer[m] for m in manufacturers},
        "model_info": model_info,
        "locations": locations,
        "ncr_cities": config.NCR_CITIES,
        "fuel_levels": config.FUEL_LEVELS,
        "transmission_levels": config.TRANSMISSION_LEVELS,
        "body_levels": config.BODY_LEVELS,
        "condition_levels": config.CONDITION_LEVELS,
        "accident_levels": config.ACCIDENT_LEVELS,
        "trim_levels": config.TRIM_LEVELS,
        "km_q99_by_age": {str(k): int(v) for k, v in sorted(km_q99_by_age.items())},
        "km_median_by_age": {str(k): int(v) for k, v in sorted(km_median_by_age.items())},
        "defaults": {
            "Manufacturer": "Maruti Suzuki",
            "Model": "Swift",
            "Variant_Trim": "Mid",
            "Body_Type": default_model["body_types"][0],
            "Year_Purchased": default_manufacture_year,
            "Manufacturing_Year": default_manufacture_year,
            "Fuel_Type": default_fuel,
            "Transmission": default_transmission,
            "Kilometres_Driven": 50_000,
            "Number_of_Previous_Owners": 1,
            "Current_Condition": "Good",
            "Location": "Pune",
            "Accident_History": "No Accident",
            "Engine_Displacement_cc": int(default_cc),
            "Original_Purchase_Price_INR": 0,
        },
    }


def main() -> None:
    """CLI entry point: build and save ``artifacts/catalog_options.json``."""
    config.ensure_dirs()
    df = data.load_raw_data()
    data.validate_data(df)
    catalog = build_catalog(df)
    with open(config.CATALOG_PATH, "w", encoding="utf-8") as handle:
        json.dump(catalog, handle, indent=2)
    print(f"Wrote {config.CATALOG_PATH} with {len(catalog['model_info'])} models.")


if __name__ == "__main__":
    main()
