"""Project-wide configuration and shared constants."""

from __future__ import annotations

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"
APP_DIR = ROOT_DIR / "app"

DATA_PATH = DATA_DIR / "indian_used_car_dataset.csv"
SPLIT_PATH = ARTIFACTS_DIR / "split_indices.npz"
TUNING_INDICES_PATH = ARTIFACTS_DIR / "tuning_indices.npy"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
CATALOG_PATH = ARTIFACTS_DIR / "catalog_options.json"
BEST_MODEL_PATH = ARTIFACTS_DIR / "best_model.json"
TEST_PREDICTIONS_PATH = ARTIFACTS_DIR / "test_predictions.parquet"
BIAS_REPORT_PATH = ARTIFACTS_DIR / "bias_report.json"
CALIBRATION_PATH = ARTIFACTS_DIR / "calibration.json"

SEED = 42
REF_YEAR = 2026

TARGET = "Resale_Price_INR"
ID_COLUMNS = ["Car_ID"]
LEAKAGE_COLUMNS = ["Initial_Ex_Showroom_Price_INR"]
OPTIONAL_PRICE_COLUMN = "Original_Purchase_Price_INR"
USE_ORIGINAL_PRICE = False

MODEL_NAMES = ["ridge", "random_forest", "xgboost", "lightgbm", "catboost", "mlp"]
MODEL_LABELS = {
    "ridge": "Ridge",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "catboost": "CatBoost",
    "mlp": "MLP",
}
TREE_MODELS = ["random_forest", "xgboost", "lightgbm", "catboost"]

VALIDATION_FRACTION = 0.15
TEST_FRACTION = 0.15
N_PRICE_BINS = 20
TUNING_SAMPLE_SIZE = 100_000
N_OPTUNA_TRIALS = {
    "ridge": 40,
    "random_forest": 30,
    "xgboost": 40,
    "lightgbm": 40,
    "catboost": 30,
    "mlp": 30,
}

FUEL_LEVELS = ["Petrol", "Diesel", "CNG", "Hybrid", "Electric"]
TRANSMISSION_LEVELS = ["Manual", "Automatic"]
BODY_LEVELS = ["Hatchback", "Sedan", "SUV", "MUV", "Van"]
CONDITION_LEVELS = ["Poor", "Fair", "Good", "Excellent"]
ACCIDENT_LEVELS = ["No Accident", "Minor", "Major"]
TRIM_LEVELS = ["Base", "Mid", "High", "Top"]
NCR_CITIES = ["Delhi", "Gurugram", "Noida"]

ALLOWED_FUEL_TRANSMISSION = {
    "Petrol": {"Manual", "Automatic"},
    "Diesel": {"Manual", "Automatic"},
    "CNG": {"Manual", "Automatic"},
    "Hybrid": {"Automatic"},
    "Electric": {"Automatic"},
}

ORDINAL_MAPS = {
    "Variant_Trim": {"Base": 0, "Mid": 1, "High": 2, "Top": 3},
    "Current_Condition": {"Poor": 0, "Fair": 1, "Good": 2, "Excellent": 3},
    "Accident_History": {"No Accident": 0, "Minor": 1, "Major": 2},
}
ORDINAL_OUTPUT_NAMES = {
    "Variant_Trim": "Variant_Ord",
    "Current_Condition": "Condition_Ord",
    "Accident_History": "Accident_Ord",
}

# Raw columns required to build a prediction row. The target and identifiers are excluded.
RAW_INPUT_COLUMNS = [
    "Manufacturer",
    "Model",
    "Variant_Trim",
    "Body_Type",
    "Year_Purchased",
    "Original_Purchase_Price_INR",
    "Manufacturing_Year",
    "Fuel_Type",
    "Transmission",
    "Kilometres_Driven",
    "Number_of_Previous_Owners",
    "Current_Condition",
    "Location",
    "Accident_History",
    "Engine_Displacement_cc",
]

CATEGORICAL_FEATURES = [
    "Manufacturer",
    "Model_Key",
    "Variant_Trim",
    "Body_Type",
    "Fuel_Type",
    "Transmission",
    "Current_Condition",
    "Location",
    "Accident_History",
]
LOW_CARDINALITY_CATEGORICALS = [
    "Manufacturer",
    "Body_Type",
    "Fuel_Type",
    "Transmission",
    "Location",
]
ORDINAL_FEATURES = [
    "Variant_Trim",
    "Current_Condition",
    "Accident_History",
]
ORDINAL_NUMERIC_FEATURES = ["Variant_Ord", "Condition_Ord", "Accident_Ord"]

NUMERIC_FEATURES = [
    "Age",
    "Ownership_Age",
    "Kilometres_Driven",
    "Km_Per_Year",
    "Km_Ratio",
    "Engine_Displacement_cc",
    "Number_of_Previous_Owners",
    "Variant_Ord",
    "Condition_Ord",
    "Accident_Ord",
    "Delhi_Flag",
    "Delhi_Restricted",
]

INTERACTION_FEATURES = [
    "Age_Sq",
    "Age_x_Condition",
    "Age_x_Fuel_Petrol",
    "Age_x_Fuel_Diesel",
    "Age_x_Fuel_CNG",
    "Age_x_Fuel_Hybrid",
    "Age_x_Fuel_Electric",
]

DEFAULT_PARAMS = {
    "ridge": {},
    "random_forest": {
        "n_estimators": 250,
        "max_depth": 24,
        "min_samples_leaf": 2,
        "max_features": 0.6,
    },
    "xgboost": {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 8,
        "min_child_weight": 3,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_lambda": 1.0,
    },
    "lightgbm": {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_child_samples": 20,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_lambda": 1.0,
    },
    "catboost": {
        "iterations": 500,
        "learning_rate": 0.05,
        "depth": 7,
        "l2_leaf_reg": 3.0,
    },
    "mlp": {
        "hidden_layer_sizes": (256, 128),
        "alpha": 1e-4,
        "learning_rate_init": 1e-3,
        "batch_size": 512,
        "max_iter": 120,
    },
}


def model_path(model_name: str) -> Path:
    """Return the serialized model artifact path."""
    return ARTIFACTS_DIR / f"{model_name}.joblib"


def params_path(model_name: str) -> Path:
    """Return the tuned-parameter JSON path."""
    return ARTIFACTS_DIR / f"{model_name}_params.json"


def ensure_dirs() -> None:
    """Create the artifact directory if needed."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
