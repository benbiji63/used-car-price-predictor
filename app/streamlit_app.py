"""Streamlit dashboard for synthetic Indian used-car resale price prediction."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config, metrics, predict  # noqa: E402

st.set_page_config(
    page_title="Used-Car Resale Price Predictor",
    page_icon="₹",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .big-price {font-size: 2.8rem; font-weight: 750; line-height: 1.15; color: inherit;}
    .small-note {font-size: 0.88rem; opacity: 0.72;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def cached_catalog() -> dict:
    return predict.load_catalog()


@st.cache_data(show_spinner=False)
def cached_metrics() -> dict:
    return predict.load_metrics()


@st.cache_data(show_spinner=False)
def cached_test_predictions() -> pd.DataFrame | None:
    return predict.load_test_predictions()


@st.cache_data(show_spinner=False)
def cached_bias_report() -> dict:
    return predict.load_bias_report()


@st.cache_data(show_spinner=False)
def cached_calibration() -> dict:
    return predict.load_calibration()


@st.cache_resource(show_spinner=False)
def cached_model(model_name: str) -> dict:
    return predict.load_artifact(model_name)


def _missing_artifacts() -> list[str]:
    missing = []
    if not config.CATALOG_PATH.exists():
        missing.append(f"{config.CATALOG_PATH} — run `python -m src.build_catalog_options`")
    if not config.METRICS_PATH.exists():
        missing.append(f"{config.METRICS_PATH} — run notebooks 01–06")
    if not config.BEST_MODEL_PATH.exists():
        missing.append(f"{config.BEST_MODEL_PATH} — run notebook 07")
    for name in config.MODEL_NAMES:
        if not config.model_path(name).exists():
            missing.append(f"{config.model_path(name)} — run notebook for {name}")
    return missing


missing = _missing_artifacts()
if missing:
    st.error("The dashboard is missing required artifacts. Please run the training notebooks first.")
    st.code("\n".join(missing))
    st.stop()

catalog = cached_catalog()
metrics_data = cached_metrics()
test_predictions = cached_test_predictions()
bias_data = cached_bias_report()
calibration_data = cached_calibration()
best_info = predict.read_best_model()
best_key = str(best_info["model_name"])
defaults = catalog["defaults"]

st.title("Used-Car Resale Price Predictor")
st.caption(
    "Synthetic Indian used-car dataset · 500,000 rows · predictions update live · "
    "all inputs are in the main page (no sidebar)"
)

# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
manufacturers = catalog["manufacturers"]
if "manufacturer_choice" not in st.session_state or st.session_state["manufacturer_choice"] not in manufacturers:
    st.session_state["manufacturer_choice"] = defaults["Manufacturer"]

with st.container(border=True):
    st.subheader("Car details")
    c1, c2, c3, c4 = st.columns(4)
    manufacturer = c1.selectbox("Manufacturer", manufacturers, key="manufacturer_choice")
    models_for_manufacturer = catalog["models_by_manufacturer"][manufacturer]
    if "model_choice" not in st.session_state or st.session_state["model_choice"] not in models_for_manufacturer:
        st.session_state["model_choice"] = (
            defaults["Model"] if defaults["Model"] in models_for_manufacturer else models_for_manufacturer[0]
        )
    model = c2.selectbox("Model", models_for_manufacturer, key="model_choice")
    info = catalog["model_info"][f"{manufacturer}|{model}"]
    body = info["body_types"][0]
    c3.text_input("Body type (auto-filled)", value=body, disabled=True)

    trims = info["trim_levels"] or catalog["trim_levels"]
    if "trim_choice" not in st.session_state or st.session_state["trim_choice"] not in trims:
        st.session_state["trim_choice"] = defaults["Variant_Trim"] if defaults["Variant_Trim"] in trims else trims[0]
    trim = c4.selectbox("Variant / trim", trims, key="trim_choice")

    c1, c2, c3, c4 = st.columns(4)
    fuels = info["fuels"]
    if "fuel_choice" not in st.session_state or st.session_state["fuel_choice"] not in fuels:
        st.session_state["fuel_choice"] = defaults["Fuel_Type"] if defaults["Fuel_Type"] in fuels else fuels[0]
    fuel = c1.selectbox("Fuel type", fuels, key="fuel_choice")

    transmissions = info["transmissions_by_fuel"][fuel]
    if "transmission_choice" not in st.session_state or st.session_state["transmission_choice"] not in transmissions:
        st.session_state["transmission_choice"] = (
            defaults["Transmission"] if defaults["Transmission"] in transmissions else transmissions[0]
        )
    transmission = c2.selectbox("Transmission", transmissions, key="transmission_choice")

    cc_key = f"{fuel}|{transmission}"
    cc_options = info["displacements_by_fuel_transmission"][cc_key]
    if "cc_choice" not in st.session_state or st.session_state["cc_choice"] not in cc_options:
        st.session_state["cc_choice"] = cc_options[0]
    if fuel == "Electric":
        cc = 0
        c3.number_input("Engine displacement (cc)", value=0, disabled=True)
    else:
        cc = c3.selectbox("Engine displacement (cc)", cc_options, key="cc_choice")

    year_options = list(range(info["year_min"], info["year_max"] + 1))
    default_my = defaults["Manufacturing_Year"]
    if default_my not in year_options:
        default_my = max(year_options[0], min(2021, year_options[-1]))
    if "my_choice" not in st.session_state or st.session_state["my_choice"] not in year_options:
        st.session_state["my_choice"] = default_my
    if len(year_options) == 1:
        manufacturing_year = year_options[0]
        c4.text_input("Manufacturing year", value=str(manufacturing_year), disabled=True)
    else:
        manufacturing_year = c4.select_slider("Manufacturing year", options=year_options, key="my_choice")

with st.container(border=True):
    st.subheader("Usage and history")
    c1, c2, c3, c4 = st.columns(4)
    year_purchase_options = list(range(manufacturing_year, config.REF_YEAR + 1))
    default_yp = max(manufacturing_year, int(defaults["Year_Purchased"]))
    if "yp_choice" not in st.session_state or st.session_state["yp_choice"] not in year_purchase_options:
        st.session_state["yp_choice"] = default_yp
    if len(year_purchase_options) == 1:
        year_purchased = year_purchase_options[0]
        c1.text_input("Year purchased", value=str(year_purchased), disabled=True)
    else:
        year_purchased = c1.select_slider("Year purchased", options=year_purchase_options, key="yp_choice")

    if "km_num" not in st.session_state:
        st.session_state["km_num"] = int(defaults["Kilometres_Driven"])
    if "km_slider" not in st.session_state:
        st.session_state["km_slider"] = int(defaults["Kilometres_Driven"])
    if "km" not in st.session_state:
        st.session_state["km"] = int(defaults["Kilometres_Driven"])

    def _km_from_number() -> None:
        value = int(st.session_state["km_num"])
        st.session_state["km"] = value
        st.session_state["km_slider"] = value

    def _km_from_slider() -> None:
        value = int(st.session_state["km_slider"])
        st.session_state["km"] = value
        st.session_state["km_num"] = value

    c2.number_input(
        "Kilometres driven",
        min_value=300,
        max_value=400_000,
        step=1_000,
        key="km_num",
        on_change=_km_from_number,
    )
    c3.slider(
        "Kilometres driven (slider)",
        min_value=300,
        max_value=400_000,
        step=1_000,
        key="km_slider",
        on_change=_km_from_slider,
    )
    owners = c4.slider("Previous owners", min_value=0, max_value=5, value=int(defaults["Number_of_Previous_Owners"]), step=1)

    c1, c2, c3, c4 = st.columns(4)
    condition = c1.selectbox("Current condition", catalog["condition_levels"], index=catalog["condition_levels"].index(defaults["Current_Condition"]))
    accident = c2.selectbox("Accident history", catalog["accident_levels"], index=catalog["accident_levels"].index(defaults["Accident_History"]))
    location = c3.selectbox("City", catalog["locations"], index=catalog["locations"].index(defaults["Location"]))
    c4.text_input("Age (derived)", value=f"{config.REF_YEAR - manufacturing_year} years", disabled=True)

kilometres = int(st.session_state["km"])
age = config.REF_YEAR - manufacturing_year

ncr_restriction = (
    location in catalog["ncr_cities"]
    and (
        (fuel == "Diesel" and age >= 10)
        or (fuel == "Petrol" and age >= 15)
    )
)
if ncr_restriction:
    st.warning(
        f"Delhi-NCR restriction applies: this {fuel.lower()} car is {age} years old in {location}. "
        "The model should apply a substantial resale-price penalty."
    )

q99 = catalog["km_q99_by_age"].get(str(age))
if q99 is not None and kilometres > q99:
    st.info(
        f"Out-of-distribution warning: {kilometres:,} km is above the training 99th percentile "
        f"({int(q99):,} km) for a {age}-year-old car. The estimate is less reliable."
    )

row = pd.DataFrame(
    [
        {
            "Manufacturer": manufacturer,
            "Model": model,
            "Variant_Trim": trim,
            "Body_Type": body,
            "Year_Purchased": int(year_purchased),
            "Original_Purchase_Price_INR": 0.0,
            "Manufacturing_Year": int(manufacturing_year),
            "Fuel_Type": fuel,
            "Transmission": transmission,
            "Kilometres_Driven": kilometres,
            "Number_of_Previous_Owners": int(owners),
            "Current_Condition": condition,
            "Location": location,
            "Accident_History": accident,
            "Engine_Displacement_cc": int(cc),
        }
    ]
)

# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------
with st.container(border=True):
    st.subheader("Model selection")
    model_options: list[tuple[str, str]] = []
    for name in config.MODEL_NAMES:
        label = config.MODEL_LABELS[name]
        display = f"⭐ {label}" if name == best_key else label
        model_options.append((display, name))
    model_options.append(("Ensemble (average)", "ensemble"))
    labels = [label for label, _ in model_options]
    best_index = [key for _, key in model_options].index(best_key)
    selection_label = st.radio("Choose a model", labels, index=best_index, horizontal=True)
    selection = dict(model_options)[selection_label]

    selected_models = list(config.MODEL_NAMES)
    if selection == "ensemble":
        selected_models = st.multiselect(
            "Models in the ensemble",
            config.MODEL_NAMES,
            default=list(config.MODEL_NAMES),
            format_func=lambda name: config.MODEL_LABELS[name],
        )
        if not selected_models:
            st.warning("Select at least one model for the ensemble.")

    ensemble_metrics = None
    if selection == "ensemble" and selected_models:
        ensemble_metrics = predict.ensemble_metrics_from_test(selected_models, test_predictions)
        if ensemble_metrics is not None:
            st.caption(
                f"Ensemble test metrics: R²(log)={ensemble_metrics['r2_log']:.4f} · "
                f"RMSE(log)={ensemble_metrics['rmse_log']:.4f} · "
                f"MAPE={ensemble_metrics['mape']:.2f}% · Median APE={ensemble_metrics['median_ape']:.2f}%"
            )
        else:
            st.caption("Ensemble subset metrics are unavailable because test predictions have not been saved yet.")

    has_calibration = bool(calibration_data.get("factors"))
    apply_calibration = st.checkbox(
        "Apply saved bias calibration",
        value=False,
        disabled=not has_calibration,
        help=(
            "Multiplies each model's prediction by the factor in artifacts/calibration.json. "
            "Run `python -m scripts.calibrate_models` to create or refresh it."
        ),
    )
    if not has_calibration:
        st.caption(
            "No calibration factors found. Run `python -m scripts.bias_report` to inspect bias, "
            "then `python -m scripts.calibrate_models` to fit factors."
        )
    elif apply_calibration:
        st.caption("Leaderboard and headline test metrics below remain uncalibrated.")

# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
with st.container(border=True):
    st.subheader("Prediction")
    if selection == "ensemble":
        if selected_models:
            pred_inr = float(predict.predict_ensemble(row, selected_models, calibrated=apply_calibration)[0])
            if ensemble_metrics is not None:
                rmse_log = float(ensemble_metrics["rmse_log"])
            else:
                rmse_log = float(np.mean([metrics_data[name]["test"]["rmse_log"] for name in selected_models]))
            explanation = f"Average of {len(selected_models)} models in log space: " + ", ".join(
                config.MODEL_LABELS[name] for name in selected_models
            )
        else:
            pred_inr = float("nan")
            rmse_log = 0.1
            explanation = "No models selected."
    else:
        pred_inr = float(predict.predict_inr(selection, row, calibrated=apply_calibration)[0])
        rmse_log = float(metrics_data[selection]["test"]["rmse_log"])
        star = "⭐ " if selection == best_key else ""
        explanation = f"{star}{config.MODEL_LABELS[selection]} (test RMSE log={rmse_log:.4f}, MAPE={metrics_data[selection]['test']['mape']:.2f}%)"

    if np.isfinite(pred_inr):
        low, high = predict.prediction_range(pred_inr, rmse_log)
        st.markdown(f"<div class='big-price'>{metrics.format_inr(pred_inr)}</div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Low estimate", metrics.format_inr(float(low)))
        c2.metric("High estimate", metrics.format_inr(float(high)))
        c3.metric("Test RMSE (log)", f"{rmse_log:.4f}")
        st.caption(f"Estimated range: ±1.5 × test RMSE in log space. {explanation}")
    else:
        st.info("Choose at least one model to generate a prediction.")

# ---------------------------------------------------------------------------
# Compare all models
# ---------------------------------------------------------------------------
with st.container(border=True):
    st.subheader("Compare all models")
    for model_name in config.MODEL_NAMES:
        cached_model(model_name)
    compare_rows = []
    for name in config.MODEL_NAMES:
        value = float(predict.predict_inr(name, row, calibrated=apply_calibration)[0])
        compare_rows.append(
            {
                "Model": config.MODEL_LABELS[name],
                "Prediction": value,
                "Test MAPE": float(metrics_data[name]["test"]["mape"]),
                "Test RMSE(log)": float(metrics_data[name]["test"]["rmse_log"]),
                "Selected": "✓" if selection == name else "",
                "Star": "⭐" if name == best_key else "",
            }
        )
    ensemble_all_pred = float(predict.predict_ensemble(row, list(config.MODEL_NAMES), calibrated=apply_calibration)[0])
    compare_rows.append(
        {
            "Model": "Ensemble (all)",
            "Prediction": ensemble_all_pred,
            "Test MAPE": float(metrics_data.get("ensemble_average", {}).get("test", {}).get("mape", np.nan)),
            "Test RMSE(log)": float(metrics_data.get("ensemble_average", {}).get("test", {}).get("rmse_log", np.nan)),
            "Selected": "✓" if selection == "ensemble" and set(selected_models) == set(config.MODEL_NAMES) else "",
            "Star": "",
        }
    )
    compare = pd.DataFrame(compare_rows).sort_values("Prediction", ascending=False)
    chart = compare.assign(
        Highlight=np.where(compare["Selected"] == "✓", "Selected", "Other")
    )
    st.bar_chart(chart, x="Model", y="Prediction", color="Highlight", stack=False)
    display_table = compare.copy()
    display_table["Prediction"] = display_table["Prediction"].map(metrics.format_inr)
    display_table["Test MAPE"] = display_table["Test MAPE"].map(lambda x: f"{x:.2f}%" if pd.notna(x) else "—")
    display_table["Test RMSE(log)"] = display_table["Test RMSE(log)"].map(lambda x: f"{x:.4f}" if pd.notna(x) else "—")
    st.dataframe(display_table[["Model", "Prediction", "Test MAPE", "Test RMSE(log)", "Selected", "Star"]], width="stretch", hide_index=True)

# ---------------------------------------------------------------------------
# What-if analysis
# ---------------------------------------------------------------------------
with st.container(border=True):
    with st.expander("What-if analysis", expanded=False):
        tabs = st.tabs(["Price vs kilometres", "Price vs age"])

        def _batch_predict(frame: pd.DataFrame) -> np.ndarray:
            if selection == "ensemble":
                if not selected_models:
                    return np.full(len(frame), np.nan)
                return predict.predict_ensemble(frame, selected_models, calibrated=apply_calibration)
            return predict.predict_inr(selection, frame, calibrated=apply_calibration)

        with tabs[0]:
            km_values = np.linspace(5_000, 300_000, 40).round().astype(int)
            km_batch = pd.concat([row] * len(km_values), ignore_index=True)
            km_batch["Kilometres_Driven"] = km_values
            km_pred = _batch_predict(km_batch)
            km_frame = pd.DataFrame({"Kilometres driven": km_values, "Predicted price": km_pred}).set_index("Kilometres driven")
            st.line_chart(km_frame)
            st.caption("All other inputs are held fixed; only kilometres driven changes.")

        with tabs[1]:
            age_values = np.arange(0, 18)
            age_batch = pd.concat([row] * len(age_values), ignore_index=True)
            age_batch["Manufacturing_Year"] = config.REF_YEAR - age_values
            age_batch["Year_Purchased"] = np.maximum(age_batch["Year_Purchased"], age_batch["Manufacturing_Year"])
            age_pred = _batch_predict(age_batch)
            age_frame = pd.DataFrame({"Age (years)": age_values, "Predicted price": age_pred}).set_index("Age (years)")
            st.line_chart(age_frame)
            st.caption("Year purchased is raised to the manufacturing year when needed, so the row stays valid.")

# ---------------------------------------------------------------------------
# Model performance
# ---------------------------------------------------------------------------
with st.container(border=True):
    with st.expander("Model performance (test set)", expanded=False):
        rows = []
        for name in config.MODEL_NAMES:
            payload = metrics_data.get(name, {})
            test_metrics = payload.get("test", {})
            if not test_metrics:
                continue
            bias_stats = bias_data.get("global", {}).get(name, {})
            factor = calibration_data.get("factors", {}).get(name)
            rows.append(
                {
                    "Model": ("⭐ " if name == best_key else "") + config.MODEL_LABELS[name],
                    "R² (log)": f"{test_metrics['r2_log']:.5f}",
                    "RMSE (log)": f"{test_metrics['rmse_log']:.5f}",
                    "MAE (INR)": metrics.format_inr(test_metrics["mae_inr"]),
                    "RMSE (INR)": metrics.format_inr(test_metrics["rmse_inr"]),
                    "MAPE": f"{test_metrics['mape']:.2f}%",
                    "Median APE": f"{test_metrics['median_ape']:.2f}%",
                    "Mean signed bias": (
                        f"{bias_stats['mean_signed_pct']:+.2f}%"
                        if bias_stats
                        else "—"
                    ),
                    "Calib factor": f"{factor:.4f}" if factor is not None else "—",
                    "Train time": f"{payload.get('train_time_s', float('nan')):.1f}s",
                }
            )
        for key, label in [("ensemble_average", "Ensemble (all)"), ("ensemble_top3", "Ensemble (top-3)")]:
            payload = metrics_data.get(key, {})
            test_metrics = payload.get("test", {})
            if test_metrics:
                rows.append(
                    {
                        "Model": label,
                        "R² (log)": f"{test_metrics['r2_log']:.5f}",
                        "RMSE (log)": f"{test_metrics['rmse_log']:.5f}",
                        "MAE (INR)": metrics.format_inr(test_metrics["mae_inr"]),
                        "RMSE (INR)": metrics.format_inr(test_metrics["rmse_inr"]),
                        "MAPE": f"{test_metrics['mape']:.2f}%",
                        "Median APE": f"{test_metrics['median_ape']:.2f}%",
                        "Train time": "—",
                    }
                )
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

st.caption(
    "Synthetic data notice: these models learn the rule-based generator's patterns, not the real Indian used-car market. "
    "The dashboard reads the winner dynamically from artifacts/best_model.json."
)
